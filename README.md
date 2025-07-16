# NTL Detection

This repository contains the codebase and structure for our participation in the **Non-Technical Loss (NTL) Detection Challenge** by HEDNO Greece. The challenge focused on identifying fraudulent or anomalous electricity consumption using time-series and structured customer data.

## Data Availability

Due to confidentiality restrictions, the dataset used in this project **cannot be shared**. It was provided exclusively for the purposes of the challenge. 

However, to support transparency and reproducibility, we provide below a complete list of the engineered features along with brief descriptions.

---

## Feature List

| Feature Name | Description |
|--------------|-------------|
| `COUNT(consumptions)` | Number of consumption records per customer |
| `MAX(consumptions.CSS_MS_HS_USE)` | Maximum energy consumption value |
| `MEAN(consumptions.CSS_MS_HS_USE)` | Mean energy consumption |
| `MIN(consumptions.CSS_MS_HS_USE)` | Minimum energy consumption |
| `SKEW(consumptions.CSS_MS_HS_USE)` | Skewness of consumption |
| `STD(consumptions.CSS_MS_HS_USE)` | Standard deviation of consumption |
| `SUM(consumptions.CSS_MS_HS_USE)` | Total energy consumption |
| `NUM_UNIQUE(consumptions.BS_RATE)` | Number of unique billing rate codes |
| `NUM_UNIQUE(consumptions.MS_METER_NBR)` | Number of unique meter IDs |
| `MODE(consumptions.DAY(MEASUREMENT_DATE))` | Most common measurement day |
| `MODE(consumptions.MONTH(MEASUREMENT_DATE))` | Most common measurement month |
| `MODE(consumptions.WEEKDAY(MEASUREMENT_DATE))` | Most common measurement weekday |
| `MODE(consumptions.YEAR(MEASUREMENT_DATE))` | Most common measurement year |
| `NUM_UNIQUE(consumptions.DAY(MEASUREMENT_DATE))` | Unique measurement days |
| `NUM_UNIQUE(consumptions.MONTH(MEASUREMENT_DATE))` | Unique measurement months |
| `NUM_UNIQUE(consumptions.WEEKDAY(MEASUREMENT_DATE))` | Unique measurement weekdays |
| `NUM_UNIQUE(consumptions.YEAR(MEASUREMENT_DATE))` | Unique measurement years |
| `COUNT(representations)` | Number of representation periods |
| `NUM_UNIQUE(representations.SUPPLIER)` | Number of unique suppliers |
| `NUM_UNIQUE(representations.SUPPLIER_TO)` | Number of unique supplier targets |
| `MODE(representations.DAY(END_DATE))` | Most common contract end day |
| `MODE(representations.MONTH(END_DATE))` | Most common contract end month |
| `MODE(representations.WEEKDAY(END_DATE))` | Most common contract end weekday |
| `MODE(representations.YEAR(END_DATE))` | Most common contract end year |
| `NUM_UNIQUE(representations.DAY(END_DATE))` | Unique contract end days |
| `NUM_UNIQUE(representations.MONTH(END_DATE))` | Unique contract end months |
| `NUM_UNIQUE(representations.WEEKDAY(END_DATE))` | Unique contract end weekdays |
| `NUM_UNIQUE(representations.YEAR(END_DATE))` | Unique contract end years |
| `COUNT(requests)` | Number of service requests |
| `NUM_UNIQUE(requests.REQUEST_TYPE)` | Unique service request types |
| `MODE(requests.DAY(REQUEST_DATE))` | Most common request day |
| `MODE(requests.MONTH(REQUEST_DATE))` | Most common request month |
| `MODE(requests.WEEKDAY(REQUEST_DATE))` | Most common request weekday |
| `MODE(requests.YEAR(REQUEST_DATE))` | Most common request year |
| `NUM_UNIQUE(requests.DAY(REQUEST_DATE))` | Unique request days |
| `NUM_UNIQUE(requests.MONTH(REQUEST_DATE))` | Unique request months |
| `NUM_UNIQUE(requests.WEEKDAY(REQUEST_DATE))` | Unique request weekdays |
| `NUM_UNIQUE(requests.YEAR(REQUEST_DATE))` | Unique request years |
| `number_of_measurements` | Number of valid daily energy measurements |
| `number_of_zeros` | Number of zero-consumption days |
| `max_energy_per_day` | Max daily energy consumption |
| `min_energy_per_day` | Min daily energy consumption |
| `mean_energy_per_day` | Mean daily energy consumption |
| `median_energy_per_day` | Median daily energy consumption |
| `std_energy_per_day` | Std. deviation of daily energy |
| `max_measurement_interval_in_days` | Max gap between measurements |
| `min_measurement_interval_in_days` | Min gap between measurements |
| `mean_measurement_interval_in_days` | Mean gap between measurements |
| `median_measurement_interval_in_days` | Median gap between measurements |
| `days_since_last_measurement` | Days since last energy measurement |
| `parno` | Contract number (encoded) |
| `xrhsh` | Usage category |
| `contract_capacity` | Capacity of the contract |
| `acct_control` | Account control flag |
| `number_of_requests` | Total number of requests |
| `target` | Binary label: 1 = NTL, 0 = normal |

### One-Hot Encoded Request Types

| Feature | Description |
|--------|-------------|
| `MODE(requests.REQUEST_TYPE)_discon` | Most common request type: Disconnection |
| `MODE(requests.REQUEST_TYPE)_newCon` | Most common request type: New connection |
| `MODE(requests.REQUEST_TYPE)_recon` | Most common request type: Reconnection |
| `MODE(requests.REQUEST_TYPE)_reprChange` | Most common request type: Representation change |
| `MODE(requests.REQUEST_TYPE)_reprPause` | Most common request type: Representation pause |
| `MODE(requests.REQUEST_TYPE)_unknown` | Most common request type: Unknown |

### One-Hot Encoded Billing Rates

These features capture the most frequent billing rate code per customer:

MODE(consumptions.BS_RATE)_[6, 7, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 33, 36, 37, 40, 41, 42, 43, 44, 52, 53, 54, 55, other]

## ✍️ License

This repository is intended for educational and research purposes only under MIT License. For questions or reuse, please contact the maintainers.
Email: npavlidi@ee.duth.gr