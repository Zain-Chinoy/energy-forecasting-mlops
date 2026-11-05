# Raw data

`continuous_dataset.csv` is the **Panama short-term electricity load forecasting** dataset.

- Source: Aguilar Madrid, Ernesto (2021), "Short-term electricity load forecasting (Panama case study)", Mendeley Data, V1, doi: 10.17632/byx7sztj59.1
- Link: https://data.mendeley.com/datasets/byx7sztj59/1
- 48,048 hourly rows, 3 Jan 2015 to 27 Jun 2020. Target: `nat_demand` (national load, MWh).
- Weather (temperature, humidity, precipitation, wind) for Tocumen, Santiago and David, plus holiday and school-period flags.
- Licence: check the licence shown on the Mendeley page and keep the citation above if you redistribute the file.

Known quirks (handled in `src/prepare_data.py` and `src/config.py`):

- `QV2M_*` is documented as "relative humidity (%)" but its values (0.01 to 0.02) are specific humidity (kg/kg).
- `Holiday_ID` is an identifier, so it is dropped (the `holiday` flag carries the information).
- Demand drops ~15% from March 2020 (COVID-19), so 2020 is kept out of training and testing.
