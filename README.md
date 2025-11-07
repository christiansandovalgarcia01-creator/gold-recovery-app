# Gold Recovery - sMAPE Project
ECHO est  activado.
App de Streamlit para predecir `rougher.output.recovery` y `final.output.recovery`.
Calcula sMAPE por objetivo y la m‚trica final 25/75.
ECHO est  activado.
## Ejecutar local
1. python -m pip install -r requirements.txt
2. python train.py
3. python -m streamlit run app.py
ECHO est  activado.
## Estructura
- app.py
- train.py
- model_rough.pkl
- model_final.pkl
- feature_list.json
- requirements.txt
- data/  (gold_recovery_train.csv, gold_recovery_test.csv, gold_recovery_full.csv)
