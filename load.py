from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.load_model('best_model.pt')
print(xgb_model)