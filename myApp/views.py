from django.shortcuts import render
from .forms import form
import pickle
import numpy as np
import os

pickle_path=os.path.join(os.getcwd(),'myapp')
# Create your views here.
def index(request):
    data_form=form(request.POST or None)
    if data_form.is_valid():
        Review=data_form.cleaned_data['Review']
        data=Review
        file = open(os.path.join(pickle_path,"model.pickle"),'rb')
        nb = pickle.load(file)
        file.close()
        file = open(os.path.join(pickle_path,"vectorizer.pickle"),'rb')
        tv = pickle.load(file)
        file.close()
        test=np.array([Review])
        test=tv.transform(test)
        res=nb.predict(test)
        if(res[0]==1):
            Review="Bad Review"
        else:
            Review="Good Review"
        return render(request,'success.html',{'Data':data,'Review':Review})

    return render(request,'index.html',{"form":data_form})