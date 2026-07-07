# `api_v2` — Mimari (FastAPI Backend)

Car Price Prediction & MLOps API — v2.0.0. Bu doküman backend'in **gerçek kodundan** çıkarılmış tam mimarisidir. Yeni web repo'sunda **`apps/api`** olarak yaşayacak (frontend `apps/web` ile birlikte), DS pipeline'ından **ayrı** ve ona **S3 artefakt sözleşmesiyle** bağlı.

---

## 1. Sistem bağlamı — nerede duruyor

```
  ┌─────────────── DS PIPELINE repo (ayrı) ───────────────┐
  scraper → build_duckdb → build_aggregates → colab_heavy_train
                    │ publish_data_to_s3 / publish_model_to_s3
                    ▼
              ┌──────────  S3 (Railway)  ──────────┐
              │  data/cars.duckdb + data/manifest.json
              │  registry.json + vN/{model.cbm, train_data.parquet,
              │                     test_data.parquet, metrics.json, shap_summary.png}
              └────────────────┬───────────────────┘
                               │ (poll-and-swap + admin sync — READ only)
  ┌──────────── WEB APP repo (bu + apps/web) ──────────────┐
  │  apps/api (BU SERVİS)  ── Volume: {VOLUME_DIR}/cars.duckdb + vN/*
  │        │ REST/JSON (CORS)
  │        ▼
  │  apps/web (Next.js)  ──► kullanıcı
  └────────────────────────────────────────────────────────┘
```

**Rol:** Bu servis *hiç veri üretmez* ve *hiç model eğitmez*. DS pipeline'ının S3'e yayınladığı **hazır artefaktları** (DuckDB + model dosyaları) **kalıcı bir volume'a** çekip, üstünden **dashboard analitiği + fiyat tahmini + drift** sunar. Böylece DS repo'suna sıfır bağımlılık.

---

## 2. Dizin yapısı

```
apps/api/                         (bugünkü api_v2/)
├─ app/
│  ├─ main.py                     FastAPI app, lifespan, router mount, CORS, static
│  ├─ api/
│  │  ├─ routes.py                9 public endpoint (frontend sözleşmesi)
│  │  └─ admin_routes.py          6 admin endpoint (/admin/*, require_admin)
│  ├─ core/
│  │  ├─ config.py                Settings (env) — tek doğru kaynak
│  │  ├─ db.py                    DuckDB read-only context manager
│  │  ├─ s3_client.py             boto3 wrapper (yalnız READ)
│  │  └─ security.py              require_admin (fail-closed auth)
│  ├─ services/
│  │  ├─ data_service.py          dashboard/options compute + cache-first (457 satır)
│  │  ├─ predict_service.py       CatBoost yükle + tahmin (train-serve parity)
│  │  ├─ data_sync_service.py     S3 poll-and-swap (cars.duckdb)
│  │  ├─ drift_service.py         KS + EMD drift
│  │  └─ admin_service.py         model sync-s3, data upload, health
│  └─ models/schemas.py           Pydantic istek/yanıt modelleri (JSON şekilleri)
├─ requirements.txt               fastapi, uvicorn, catboost, pandas, duckdb, boto3, ...
├─ railway.json                   Railway deploy (RAILPACK, uvicorn start)
└─ .env.example                   config şablonu (sırlar .env'de, GITIGNORED)
```

---

## 3. Uygulama yaşam döngüsü (`main.py`)

- **App:** `FastAPI(title="Car Price Prediction & MLOps API", version="2.0.0")`, `/docs` + `/redoc`, 5 OpenAPI tag (Model Management, Prediction, Drift Detection, Dashboard & Analytics, Admin).
- **CORS:** `allow_origins=settings.ALLOWED_ORIGINS` (**varsayılan `["*"]`**), `allow_credentials=True`, methods/headers `*`. → **Prod'da `ALLOWED_ORIGINS`'i frontend domain'ine daralt** (credentials=True + `*` tarayıcıda sorun çıkarır).
- **Static:** `app.mount("/reports", StaticFiles(directory=STATIC_DIR))` → `static_reports/` klasörü.
- **Lifespan (startup):** (1) `preload_latest_model()` best-effort (başarısızsa çökmez, ilk `/predict`'te lazy yükler); (2) `DATA_SYNC_POLL_SECONDS>0` ise `data_sync_service.poll_loop()` arka plan task'ı.
- **Lifespan (shutdown):** poll task iptal + `unload_models()` (bellek temizliği).
- **Çalıştırma:** Railway → `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

---

## 4. Veri katmanı (`core/db.py` + `data_service.py`)

- **Kaynak:** **lokal** `{VOLUME_DIR}/cars.duckdb` (S3 değil — request anında volume'dan okunur). Tablolar: `car_listings`, `dashboard_cache`, `options_cache`, `price_history`.
- **Erişim:** `get_db_connection(read_only=True)` context manager — her `with` bloğunda taze bağlantı, `finally`'de kapanır. **Pool yok**, API dosyayı asla değiştirmez. Dosya yoksa `FileNotFoundError`.
- **İki katman:** (1) **cache-first hızlı yol** — `dashboard_cache`/`options_cache` tablolarından hazır JSON payload (sadece brand/series scope'lu, aralık-filtresi yoksa; 33k satır taranmaz); (2) **canlı fallback** — `compute_dashboard`/`compute_options` DuckDB'de anlık hesaplar.
- **Zaman-scope'lama** — `_time_source(as_of, mode)`:
  - `mode="specific"` + `as_of` → sadece o gün (`search_date = $as_of`).
  - `mode="until"` + `as_of` → point-in-time: her `ad_id` için `search_date <= as_of` en son kayıt (dedup pencere sorgusu).
  - `as_of=None` → tüm snapshot'lar üzerinden güncel durum (dedup).
  - Dedup: `row_number() OVER (PARTITION BY ad_id ORDER BY search_date DESC, scraped_at DESC) = 1` → **ad_id başına tek satır**.
- **Scope filtresi** — `build_filter_clause`: `brand`, `series`, `price`, `kb_year`, `kb_mileage`, `kb_fuel` (DuckDB `$name` param; `"Tümü"` atlanır).

---

## 5. Konfigürasyon (`core/config.py`) — env değişkenleri

`Settings(BaseSettings)`, `.env`'den okur (`extra="ignore"`). **Sırlar `.env`'de, ASLA commit edilmez.**

| Env | Tip | Varsayılan | Anlam |
|---|---|---|---|
| `PROJECT_NAME` | str | "Car Price Prediction & MLOps API" | Başlık |
| `ALLOWED_ORIGINS` | list | `["*"]` | CORS — **prod'da daralt** |
| `VOLUME_DIR` | str | repo kökü (env yoksa) | Kalıcı volume kökü; veri+model tek kaynak. Railway'de `/data` mount |
| `DUCKDB_PATH` | str | `{VOLUME_DIR}/cars.duckdb` | Lokal DuckDB |
| `ADMIN_API_KEY` | str | `""` | `/admin/*` anahtarı; **boşsa admin 503 (fail-closed)** 🔒 |
| `ADMIN_IP_ALLOWLIST` | str | `""` | Virgüllü IP allowlist; boşsa kapalı |
| `MODEL_CACHE_ENABLED` | bool | **`False`** | False → her `/predict`'te modeli diskten taze yükle (RAM az, latency ↑). True → bellekte cache + startup preload |
| `RAILWAY_S3_ENDPOINT/ACCESS_KEY/SECRET_KEY/BUCKET` | str | `""` | S3 (serving store + model kaynağı) 🔒 |
| `DATA_SYNC_POLL_SECONDS` | int | `300` | S3 veri-sync poll aralığı; `0` → kapalı (veri sadece `/admin/data/upload`) |
| `DATA_S3_KEY` | str | `data/cars.duckdb` | S3'teki duckdb anahtarı |
| `DATA_MANIFEST_KEY` | str | `data/manifest.json` | S3 versiyon manifesti |

---

## 6. API sözleşmesi — public endpoint'ler (`api/routes.py`)

Frontend'in tükettiği JSON sözleşmesi. **9 endpoint** (prefix yok, root'ta):

| Method | Path | Query/Body | Yanıt | Amaç |
|---|---|---|---|---|
| GET | `/versions` | — | `list[VersionInfo]` | Model versiyonları (yeni→eski) |
| GET | `/api/shap/{version_id}` | — | PNG (`image/png`) | Versiyonun SHAP grafiği |
| POST | `/predict/{version_id}` | body `CarPredictionInput` | `PredictionResponse` | Fiyat tahmini (Q50 + Q5–Q95 + risk) |
| GET | `/drift/{ref_ver}/{curr_ver}` | `brand?` | `DriftResponse` | İki model eğitim-verisi drift'i |
| GET | `/api/data-drift` | `ref:date, curr:date, mode=specific, brand?` | `DriftResponse` | İki snapshot arası drift |
| GET | `/api/dashboard-data` | `brand?, series?, min/max_price, min/max_year, min/max_km, fuel?, as_of?, mode=until` | `DashboardResponse`* | Dashboard analitiği (KPI+grafikler) |
| GET | `/api/snapshots` | — | JSON* | Snapshot tarihleri + sayımlar |
| GET | `/api/price-history/{ad_id}` | — | JSON* | Bir ilanın fiyat/km zaman serisi |
| GET | `/api/options` | `brand?, series?, as_of?, mode=until` | `DropdownOptionsResponse` | Kaskad dropdown (marka→seri→model) |

\* **Uyarı:** `/api/dashboard-data`, `/api/snapshots`, `/api/price-history` decorator'da **`response_model` YOK** → şekil yalnız servis fonksiyonunca belirlenir (doğrulanmaz). `dashboard-data` şekli `DashboardResponse`'a uyar ama zorlanmaz; diğer ikisinin Pydantic modeli yok. `mode` varsayılanı farklı: `data-drift`=`specific`, `dashboard-data`/`options`=`until`. Hata mesajları Türkçe.

### Yanıt şekilleri (`models/schemas.py`)
- **`DashboardResponse`**: `brands[], seriesList[], kpi{total,avgPrice}, boxplotData{brand:[fiyatlar]}, scatterData{brand:[[km,fiyat]]}, lineChartData{years[],prices[]}, donutChartData[{name,value}], damageChartData[{part,degisen,boyali,lokal,value}], radarChartData{indicators[],series[]}` (radar sadece ≤5 marka).
- **`DropdownOptionsResponse`**: `{brands[], series[], models[]}` (seçilmeyen seviye `[]`).
- **`DriftResponse`**: `{results:[FeatureDriftResult{feature,drift_detected,p_value,ks_statistic,emd_score,normalized_emd,chart_data:[{bin,ref_density,curr_density}]}]}`.
- **`VersionInfo`**: `{version_id, date?}` + `extra=allow` (ek alanlar olabilir).

---

## 7. Tahmin servisi + train-serve PARITY (`predict_service.py`) ⚠️

- **Model:** CatBoost `.cbm`, `{VOLUME_DIR}/{version_id}/model.cbm`'den `load_model`. **MultiQuantile:alpha=0.05,0.5,0.95** → `predict` (1,3) döner (q05/q50/q95).
- **Versiyon:** `/predict/{version_id}` **zorunlu path param** — "latest" otomatik seçilmez. `registry.json` (volume) tarih-desc.
- **Girdi şeması (17 alan):** `CarPredictionInput` 16 alan verir + `expert_risk_score` **sunucuda türetilir** (hasar ağırlıkları `model_train.py` ile birebir). Kolonlar `df = df[model.feature_names_]` ile **modelin kendi sırasına** indekslenir (parity zorlama noktası). 9 kategorik `astype(str)`.
- **Çıktı:** `q05*0.98`, `q95*1.02` (±%2 heuristik genişletme) → `PredictionResponse{price:int(q50), price_range:{min,max,margin_percent}, version, calculated_risk_score, currency:"TL"}`.

### 🔴 KRİTİK: bu servis ESKİ model şemasına bağlı
`predict_service` bugün **eski** `pipeline/model_train.py` modelini (registry v1–v3, PostgreSQL, `segment_clean`+`expert_risk_score`+`cylinder_count`) bekliyor. **DS pipeline'daki YENİ model** (`colab_heavy_train.py`) farklı şema kullanıyor:
`CAT=[brand,series,kb_body_type,kb_drivetrain,gb_segment_imp,kb_transmission,kb_fuel]`, `NUM=[vehicle_age,gb_mileage,power_hp_val,engine_cc_val,torque_nm,count_painted,count_changed,count_local_painted,is_heavy_damaged]`, `TEXT=[model]`.
→ **Yeni modeli servise koymadan önce `CarPredictionInput` + `feature_dict` bu yeni şemaya güncellenmeli** (ör. `segment_clean`→`gb_segment_imp`, `expert_risk_score`/`cylinder_count` çıkar, `year`→`vehicle_age`, hasar sayaçları ekle). **Rebuild'in en önemli işi budur.**

### Diğer parity bayrakları (kodda mevcut)
- **`is_heavy_damaged` ölü alan** — eski eğitim onu düşürüyor; API alıyor ama reindex sessizce atıyor (tahmine etkisiz).
- **Allowlist-vs-blocklist kırılganlığı** — eğitim `SELECT *` − blocklist ile feature kuruyor; DB'ye yeni kolon girerse model feature'ı olur ama serving dict'inde olmaz → `df[feature_names_]` **request'te `KeyError` → HTTP 500**. Ortak şema/manifest yok; sadece mevcut kolonlar denk geldiği için çalışıyor.
- **`model` etiket uyuşmazlığı** — eğitimde text feature, serving'de yerel `cat_features` listesinde (kozmetik, `.cbm`'deki tip geçerli).
- **Heuristik aralık** — gösterilen min/max modelin kalibre kapsaması (registry ~%84.6) değil, ±%2 nudge'lı.
- **`MODEL_CACHE_ENABLED=False`** — her `/predict` diskten tam deserialize (latency maliyeti).

---

## 8. Veri & model artefakt sözleşmesi (S3 ↔ volume)

**Veri (cars.duckdb) — poll-and-swap** (`data_sync_service.py`): her tick `manifest.json` GET → `version` karşılaştır (`.data_version.json` volume'da). Değiştiyse: `DATA_S3_KEY`'i **aynı filesystem'de** temp'e indir → opsiyonel `sha256` doğrula → tabloları validate et → **`os.replace` atomik swap** → yeni versiyonu yaz. Hata olursa canlı DB'ye dokunulmaz. Single-flight (`asyncio.Lock`), bloklayan I/O `to_thread`'de, restart'lar idempotent.

**Model — admin sync** (`admin_service.sync_s3`): S3 `registry.json` + her versiyon için **5 dosya** çeker: `model.cbm, train_data.parquet, test_data.parquet, metrics.json, shap_summary.png` → volume `vN/`. `POST /admin/models/sync-s3` ile tetiklenir (boş `version_ids` = hepsi).

**S3 client** (`s3_client.py`): boto3, **yalnız READ** (download/read_json/list/exists). Yazma = DS pipeline'ın işi (`publish_*_to_s3.py`).

**Validasyon:** `_validate_duckdb` gerekli tabloları (`car_listings, dashboard_cache, options_cache`) + boş-olmayan `car_listings` şart koşar; geçersiz dosya canlıya geçmez.

---

## 9. Admin & güvenlik

**Admin endpoint'leri** (`/admin/*`, hepsi `require_admin`):
| Method | Path | İş |
|---|---|---|
| GET | `/admin/health` | Volume + model/data durumu |
| GET | `/admin/models` | Volume'daki versiyonlar (registry + disk + bellek) |
| POST | `/admin/models/sync-s3` | Model(ler) S3→volume + registry merge (207 kısmi hata) |
| DELETE | `/admin/models/{version_id}` | Versiyon sil (409: son versiyonu koruma) |
| POST | `/admin/models/{version_id}/preload` | Belleğe yükle (409: cache kapalıysa) |
| POST | `/admin/data/upload` | Hazır `cars.duckdb` yükle → validate → atomik swap |

**Güvenlik** (`security.py`, sırayla, **fail-closed**): (1) `ADMIN_API_KEY` boşsa **503**; (2) IP allowlist dışıysa **403** (X-Forwarded-For ilk hop); (3) rate limit **30/60s/IP → 429**; (4) `X-Admin-Key` **constant-time** karşılaştırma, hatalıysa **401**. Tüm kararlar `admin.audit` logger'a (sır loglanmaz). *Not: rate limit process-local → tek replica varsayımı.*

---

## 10. Drift (`drift_service.py`)
Sayısal kolonlarda **KS testi + EMD (Wasserstein)**. `drift_detected = p<0.05 AND normalized_emd>0.1` (anlamlılık + etki). 20-bin yoğunluk histogramı. İki giriş: `/api/data-drift` (car_listings snapshot dilimleri) ve `/drift/{ref}/{curr}` (model `train_data.parquet` çiftleri).

---

## 11. Deploy (Railway)
- `railway.json`: RAILPACK builder, `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, 1 replica (us-west2), `restartPolicy=ON_FAILURE` (max 10), sleep kapalı.
- **Volume:** `/data` mount → `VOLUME_DIR=/data`. Veri+model burada kalıcı; S3 sync/admin ile beslenir.
- **Env:** tüm `.env` değişkenleri Railway env var olarak set edilir. `ADMIN_API_KEY` + `RAILWAY_S3_*` zorunlu.

---

## 12. Rebuild notları (siteye uyum için) 🎯
Sen `api_v2`'yi yeni repo'ya `apps/api` olarak taşırken:
1. **Train-serve şemasını düzelt (öncelik):** yeni CatBoost modeli için `CarPredictionInput` + `predict_service.feature_dict`'i `colab_heavy_train.py` şemasına hizala (§7 kırmızı kutu). İdeal: model artefaktıyla **şema manifesti** yayınla (`metrics.json.features`), serving onu okusun → allowlist/blocklist kırılganlığı biter.
2. **CORS'u daralt:** `ALLOWED_ORIGINS` = frontend domain'i (credentials=True ile `*` çalışmaz).
3. **Frontend sözleşmesi:** §6 tablosu + şemalar `apps/web`'in beklediği JSON. Değişecek endpoint varsa ikisini birlikte güncelle (aynı repo avantajı).
4. **`response_model` ekle:** dashboard/snapshots/price-history için Pydantic model → tip güvenliği + otomatik OpenAPI (frontend tip üretimi).
5. **Sırlar:** `.env` gitignored; taşırken **commit etme**, Railway env var kullan. Paylaşılmış anahtarları **rotate et**.
6. **Data kaynağı:** `DATA_SYNC_POLL_SECONDS>0` + S3 manifest ile otomatik; DS pipeline `publish_data_to_s3` çalıştıkça API kendini günceller — **iki repo arası tek bağ budur**.
