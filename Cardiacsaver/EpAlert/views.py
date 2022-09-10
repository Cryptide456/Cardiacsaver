from django.shortcuts import render
import joblib

# Create your views here.
def home(request):
  return render(request, "EpAlert/home.html")

def about(request):
  return render(request, "EpAlert/about.html")

def form(request):
  return render(request, "EpAlert/form.html")

def model(request):
  return render(request, "EpAlert/model.html")

def results(request):

  #nb = joblib.load('nb_model.sav')
  rf = joblib.load('rf_model.sav')
  knn = joblib.load('knn_model.sav')
  dt = joblib.load('dt_model.sav')
  svc = joblib.load('svc_model.sav')
  #scv = joblib.load('scv_model.sav')
  ada = joblib.load('clf_model.sav')    # ada boosting ensemble
  gb = joblib.load('clf1_model.sav')    # gradient boosting ensemble

  lis = []

  # attributes must be appended inside list 'lis'
  # in the same order as in the training data set
  lis.append(request.GET['age'])
  lis.append(request.GET['sex'])
  lis.append(request.GET['cp'])
  lis.append(request.GET['resting-bps'])
  lis.append(request.GET['chol'])
  lis.append(request.GET['fasting-blood-sugar'])
  lis.append(request.GET['resting-ecg'])
  lis.append(request.GET['thalach'])
  lis.append(request.GET['exang'])
  lis.append(request.GET['oldpeak'])
  lis.append(request.GET['slope'])
  lis.append(request.GET['ca'])
  lis.append(request.GET['thal'])

  #nbans = nb.predict([lis])
  rfans = rf.predict([lis])
  knnans = knn.predict([lis])
  dtans = dt.predict([lis])
  svcans = svc.predict([lis])
  #scvans = scv.predict([lis])
  adaans = ada.predict([lis])
  gbans = gb.predict([lis])

  return render(request, "EpAlert/results.html", {'rfans':rfans, 'knnans':knnans, 'dtans':dtans, 'svcans':svcans, 'adaans':adaans, 'gbans':gbans})