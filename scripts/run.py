# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
import datetime
from helpers import *
from implementations import *


from proj1_helpers import *

# This function was not used since it does not improve the results

def drop_col(tX_starting):
    drop_columns=[]
    #for i in range(tX_starting.shape[1]):
    #    coeff=np.corrcoef(y,tX_starting[:,i])[0,1]
    #    if abs(coeff)<0.000:
    #        drop_columns.append(i)


    tX=np.delete(tX_starting,drop_columns,axis=1)
    return tX
   

def replace_nan(tX_starting,median=False):
    tX=tX_starting.copy()
    # Dummy features added corresponding to the nan pattern
    nan_position=[tX[:,[0,4,23]]!=-999][0]*1

    for col in range(tX.shape[1]):
        column=tX[:,col][tX[:,col]!=-999]
        if median==False:
            mean=column.mean()
            median=np.median(column)

        tX[:,col][tX[:,col]==-999]=median    
    
    return nan_position,tX,median

def categorical_variables(tX_starting):
    tX=tX_starting.copy()
    
    cat_variable=22
    values=[0,1,2]

    added_matrix=np.zeros([tX.shape[0],3])
    added_matrix[:,0]=np.array([tX[:,22]==0])
    added_matrix[:,1]=np.array([tX[:,22]==1])
    added_matrix[:,2]=np.array([tX[:,22]==2])
    
    tX=np.delete(tX,[22],axis=1)
    
    return added_matrix,tX

def before_poly(tX_starting,median=False):
    tX=drop_col(tX_starting)
    nan_position,tX,median=replace_nan(tX,median)
    added_matrix,tX=categorical_variables(tX)
    full_added_matrix=np.concatenate((added_matrix,nan_position),axis=1)
    return full_added_matrix,tX,median

def build_poly(tX,degree,y,prod_to_exclude=False,train=True,columns_to_consider=False,exponential=False,cross_products=False,added_matrix_for_cross=False,threshold_power=0.0,threshold_cross=0.00,exclude=False):
    # Some feature can be not considered
    if not columns_to_consider:
        columns_to_consider=range(tX.shape[1])
    # Cross products to exclude
    if not prod_to_exclude:
        prod_to_exclude=[]
    if exclude==False:
        exclude=[]
    dict_cross={}
    
    # Features to include in hte model
    columns_to_consider=[x for x in columns_to_consider if x not in exclude]
    columns_to_consider=np.array(columns_to_consider)
    # Add power of the matrix
    final_list=[]
    for i in range(2,degree+1):
        cols=columns_to_consider
        tX=np.concatenate((tX,tX[:,cols]**i),axis=1)
        if i%2==1:
            tX=np.concatenate((tX,np.sqrt(abs(tX[:,cols]**i))),axis=1)
    # Take the exponential of the features
    if exponential:
        tX=np.concatenate((tX,np.exp(tX[:,cols]/100)),axis=1)
        tX=np.concatenate((tX,np.exp(tX[:,cols]/80)),axis=1)
        tX=np.concatenate((tX,np.exp(tX[:,cols]/60)),axis=1)
        tX=np.concatenate((tX,np.exp(tX[:,cols]/50)),axis=1)
        tX=np.concatenate((tX,np.exp(tX[:,cols]/40)),axis=1)
        tX=np.concatenate((tX,np.exp(tX[:,cols]/20)),axis=1)

    # Cross products of the features
    if cross_products:
        # The dummy variables are considered for the cross products
        if added_matrix_for_cross.any():
            # Add to columns to consider
            for i in range(tX.shape[1],tX.shape[1]+added_matrix_for_cross.shape[1]):
                columns_to_consider=np.append(columns_to_consider,i)
            # Concatenate
            tX=np.concatenate((tX,added_matrix_for_cross),axis=1)
            final_list.append(tX)
        start_cross=tX.shape[1]
        for i,col1 in enumerate(columns_to_consider):
            for j,col2 in enumerate(columns_to_consider):
                if j>i and (i,j) not in prod_to_exclude:
                    if train:
                        prod=tX[:,col1]*tX[:,col2]
                        corr=np.corrcoef(prod,y)[0,1]
                        if abs(corr)>threshold_cross:
                            final_list.append(prod.reshape([prod.shape[0],1]))

                            #print(start_cross,type(start_cross))
                            dict_cross[start_cross]=tuple([i,j])
                            start_cross+=1
                        else:
                            prod_to_exclude.append((i,j))
                    else:
                        prod=tX[:,col1]*tX[:,col2]
                        final_list.append(prod.reshape([prod.shape[0],1]))
        final_tuple=tuple(final_list)
        tX=np.concatenate(final_tuple,axis=1)
    return tX,prod_to_exclude,dict_cross

# If train==False the mean and the std are the ones computed in the train matrix.
def normalize(tX,mean=False,std=False,train=False):
    if train:
        mean=np.sum(tX,axis=0)/tX.shape[0]
        std=np.sqrt(np.sum(tX**2,axis=0)/tX.shape[0])
    tX=(tX-mean)/std
    if train:
        return tX,mean,std
    else:
        return tX

def add_ones(tX_starting):
    ones=np.ones(tX_starting.shape[0]).reshape([tX_starting.shape[0],1])
    tX=np.concatenate((tX_starting,ones),axis=1)
    return tX

def process_data(tX_starting,y,prod_to_exclude=False,train=True,mean=False,std=False,median=False,exclude=False):
    full_added_matrix,tX,median=before_poly(tX_starting,median)
    tX,prod_to_exclude,dict_cross=build_poly(tX,14,y,exclude=exclude,train=train,prod_to_exclude=prod_to_exclude,exponential=True,cross_products=True,added_matrix_for_cross=full_added_matrix,threshold_cross=0.0)
    if train:
        tX,mean,std=normalize(tX,train=True)
    else:
        tX=normalize(tX,mean,std,train=False)
    tX=add_ones(tX)
    
    if train:
        return tX,prod_to_exclude,mean,std,median,dict_cross
    else:
        return tX











DATA_TRAIN_PATH = '../../train.csv' # TODO: download train data and supply path here 
y_starting, tX_starting, ids = load_csv_data(DATA_TRAIN_PATH,sub_sample=False)


print('Data read')

exc=[]
tX=tX_starting.copy()
y=y_starting.copy()
tX,prod_to_exclude,mean,std,median,dict_cross=process_data(tX,y,train=True,exclude=exc)
print('Data processed')

lamb=-5e-05
w,loss=ridge_regression(y,tX,lamb)
print('Got weights')

DATA_TEST_PATH = '../../test.csv' 
_, tX_final_test, ids_test = load_csv_data(DATA_TEST_PATH)
print('Test data read')

tX_final=process_data(tX_final_test,y=y,prod_to_exclude=prod_to_exclude,mean=mean,std=std,median=median,train=False)
print('Test data processed')


OUTPUT_PATH = 'predictions.csv' 
y_pred = predict_labels(w, tX_final)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print('Output to '+OUTPUT_PATH)



















