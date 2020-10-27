from sklearn.model_selection import ShuffleSplit,StratifiedShuffleSplit
import numpy as np
import itertools
from .run import run_params
from .kmer_chemistry import get_AX
from .nn_model import initialize_model, initialize_filters
from multiprocessing import Pool, Manager
import os
import boto3
'''
Script will generate cross validation folds in two ways:
1 - Random 90-10, 80-20, ... splits to test how little information we need to
    train on.
2 - CV splits with missing bases in the training, which are present in the test
    set
'''

def gen_all_kmers(alphabet=['A','T','C','G','M'], repeat=6):
    '''
    alphabet: bases to be constructed into kmers
    repeat: length of kmer
    '''
    combinations = list(itertools.product(alphabet,repeat=repeat))
    return list(map(lambda x:''.join(x), combinations))

def kmer_parser(fn, exclude_base=None):
    '''
    Function parses kmer file and returns

    Parameters
    ----------
    fn: str
        path to file
    exclude_base : str
        base to exclude from kmer_list. The base selected will
        be removed regardless its position. All kmers containing
        that base will not be returned

    Returns
    ----------
    kmer_list: array, list of kmers in the order they appear in the file
    pA_list: array, list of pA values (floats) in the same order
    '''
    kmer_list = []
    pA_list = []
    label_list = []
    with open(fn ,'r') as f:
        lines = f.readlines()
        for line in lines:
            if '\t' in line:
                line = line.split('\t')
            elif ' ' in line:
                line = line.split(' ')
            if len(line) > 2:
                label = int(line[2].strip())
                label_list += [label]
            kmer = str(line[0]).strip()
            pA = float(line[1].strip())
            if exclude_base is not None:
                if exclude_base in kmer:
                    continue
            kmer_list += [kmer]
            pA_list += [pA]
    
    if len(label_list) == 0:
        label_list = None
    
        return np.array(kmer_list), np.array(pA_list), label_list
    else:
        return np.array(kmer_list), np.array(pA_list), np.array(label_list)

def kmer_parser_enc(fn):
    '''
    Function parses kmer file and returns an encoded version of the bases
    A:0, T:1, C:2, G:3, M:4, Q:5

    Parameters
    ----------
    fn: str, path to file

    Returns
    ----------
    kmer_list: array, list of kmers in the order they appear in the file
    pA_list: array, list of pA values (floats) in the same order
    '''
    enc = {'A':0,'T':1, 'C':2, 'G':3, 'M':4, 'Q':5}
    
    kmer_list = []
    pA_list = []
    with open(fn ,'r') as f:
        lines = f.readlines()
        for line in lines:
            if '\t' in line:
                line = line.split('\t')
            elif ' ' in line:
                line = line.split(' ')
            kmer = list(str(line[0]).strip())
            
            pA = float(line[1].strip())
            for i, base in enumerate(kmer):
                kmer[i] = enc[base]
            
            kmer_list += [kmer]
            pA_list += [pA]
    
    #kmer_mat = np.vstack(kmer_list)
    
    return np.array(kmer_list), np.array(pA_list)

def cg_mg_combine():
    '''
    Function combines native and methylated kmers
    
    Returns
    -------
    all_data: np matrix - all kmers both native and methylated
    all_pA: np matrix - all pA measures from both native and methylated kmers
    all_labels = np matrix - denotes what kind of kmer exists in each index
    '''
    cg = "./ont_models/r9.4_180mv_450bps_6mer_DNA.model"
    mg = "./ont_models/r9.4_450bps.mpg.6mer.template.model"

    cg_kmer, cg_pA = kmer_parser(cg)
    mg_kmer, mg_pA = kmer_parser(mg)

    cg_kmer = cg_kmer.reshape(-1,1)
    cg_pA = cg_pA.reshape(-1,1)
    mg_kmer = mg_kmer.reshape(-1,1)
    mg_pA = mg_pA.reshape(-1,1)

    cg_labels = np.array(cg_kmer.shape[0]*[0]).reshape(-1,1)
    mg_labels = np.array(mg_kmer.shape[0]*[1]).reshape(-1,1)


    all_data = np.vstack([cg_kmer, mg_kmer])
    all_pA = np.vstack([cg_pA, mg_pA])
    all_labels = np.vstack([cg_labels, mg_labels])

    return all_data, all_pA, all_labels

def cv_folds(X, Y,labels=None, folds=5, test_sizes = np.arange(0.1,1,0.1)):
    '''
    Parameters
    -----------
    X : array
        list of samples

    Y : array
         list of values to predict

    labels : array
         list of labels to be used for stratified split

    folds : int
         number of CV folds to be made

    test_sizes : array
        Array to test sizes

    Returns
    -----------
    test_size : float

    kmer_train_mat : mat
        shape(folds, train_size) for each train/test split

    kmer_test_mat : mat
        shape(folds, train_size) for each train/test split

    pA_train_mat : mat
        shape(folds, test_size) for each train/test split

    pA_test_mat : mat
        shape(folds, test_size) for each train/test split

    '''

    #test_sizes = np.arange(0,1,0.1)[1:] #excluding zero
    #test_sizes = [0.9,0.75, 0.5, 0.25, 0.1]
    

    for test_size in test_sizes:

        # The following matrices contain the train/test kmer and corresponding pA values
        # for the folds produced. Shape is (folds, train/test size)
        kmer_train_mat = []
        kmer_test_mat = []

        pA_train_mat = []
        pA_test_mat = []
        
        if labels is not None:
            splitter = StratifiedShuffleSplit(n_splits=folds, test_size=test_size, random_state=42).split(X, labels) 
        else:
            splitter = ShuffleSplit(n_splits=folds, test_size=test_size, random_state=42).split(X)

        for train_idx, test_idx in (splitter):
            x_train = X[train_idx]
            x_test = X[test_idx]

            y_train = Y[train_idx]
            y_test = Y[test_idx]

            if all(isinstance(kmer, str) for kmer in x_train):
                print("worked")
                x_train = x_train.flatten()
                x_test = x_test.flatten()
                
                kmer_train_mat += [x_train]
                kmer_test_mat += [x_test]
                
            else:
                
                kmer_train_mat += [np.vstack(x_train)]
                kmer_test_mat += [np.vstack(x_test)]

            pA_train_mat +=[y_train]
            pA_test_mat += [y_test]
        
        pA_train_mat = np.vstack(pA_train_mat)
        pA_test_mat = np.vstack(pA_test_mat)
 
        if all(isinstance(kmer, str) for kmer in x_train): 
            kmer_train_mat = np.vstack(kmer_train_mat)
            kmer_test_mat = np.vstack(kmer_test_mat)
        
        else:
            kmer_train_mat = np.array(kmer_train_mat)
            kmer_test_mat = np.array(kmer_test_mat)

        yield test_size,kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat


def base_folds(kmer_list, pA_list, bases):

    '''
    Function generates train test splits based on DNA base position on the kmer
    For example:
        A train split will contain no A bases in the first position.
        The test split will contain all kmers with A in the first position
        And so on...

    Each train/test matrix will have shape(n_bases[4], train/test size)
    The order of the bases, meaning rows, in these matrices is [A,T,C,G].
    Six matrices will be produced with size (4,train/test size)

    For the train matrices, the base is ABSENT
    For the test matrices, the base in PRSENT
    '''

    #bases = ['A', 'T', 'C', 'G']
    positions = np.arange(0,len(kmer_list[0])) # assuming all kmers in kmer_list are of equal length

    new_kmer_list = []
    for kmer in kmer_list:
        new_kmer_list += [list(kmer)]
    new_kmer_list = np.vstack(new_kmer_list)

    for pos in positions:
        print("examining position:",pos)
        pos_bases = new_kmer_list[:,pos]

        kmer_train_mat = []
        kmer_test_mat = []

        pA_train_mat = []
        pA_test_mat = []

        for base in bases:
            # print("examining base:",base)
            train_idx = np.argwhere(pos_bases!=base).flatten() #base in this pos absent from training
            test_idx = np.argwhere(pos_bases==base).flatten() #base in this pos present in testing

            kmer_train_mat += [kmer_list[train_idx]]
            kmer_test_mat += [kmer_list[test_idx]]

            pA_train_mat += [pA_list[train_idx]]
            pA_test_mat += [pA_list[test_idx]]

            # print(np.argwhere(pos_bases==base).flatten().shape) #test
            # print(np.argwhere(pos_bases!=base).flatten().shape) #train

        kmer_train_mat = np.vstack(kmer_train_mat)
        kmer_test_mat = np.vstack(kmer_test_mat)

        pA_train_mat = np.vstack(pA_train_mat)
        pA_test_mat = np.vstack(pA_test_mat)

        # print(kmer_train_mat.shape)
        # print(kmer_test_mat.shape)
        #
        # print(pA_train_mat.shape)
        # print(pA_test_mat.shape)

        yield pos, kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat

class GPUGSCV:
    
    '''
    GPU enbaled grid search cross validation
    A parameter dict is provided and for each combination of parameters k-fold cv is done
    If multiple gpus are available, each of the parameter combinations will be done on a separate gpu
    '''

    def __init__(self,model,param_dict, cv=10, n_gpus=1,res_fn=None, n_type="DNA"):
        self.model = model # function that initializes keras model
        self.param_dict = param_dict
        self.original_keys = self.param_dict.keys()
        self.cv = cv
        self.n_gpus = n_gpus
        self.res_fn = res_fn # path to save results, if selected res_dict will be saved after each parameter combination is done
        self.combined_params = self._combine_params
        self.n_type = n_type

        try:
            s3out = str(os.environ['S3OUT'])
            local_out = str(os.environ['MYOUT'])
            prp = str(os.environ['PRP'])

            session = boto3.session.Session(profile_name="default")
            bucket = session.resource("s3", endpoint_url=prp).Bucket("stuartlab")
            bucket.download_file(s3out+res_fn,'.'+local_out+res_fn )
            self.res_dict = np.load('.'+local_out+res_fn, allow_pickle=True).item()
        except:
            self.res_dict = {}

        self.best_params = None
        self.best_score = None
 
        self.cv_results = None

    
    @property
    def _combine_params(self):
        '''
        Method makes all combinations of provided list of values for all parameters
        '''
        list_values = self.param_dict.values()
        
        combined_values_list = list(itertools.product(*list_values))
        
        return combined_values_list

    def fit(self, kmer_list,pA_list, labels=None):
       
        manager = Manager()
        best_score_params = manager.list() #[best_score, best_params]
        best_score_params.append(None) # best_score float
        best_score_params.append(None) # best_params dict
        gpu_n = np.arange(self.n_gpus)
        avail_gpus = manager.list(gpu_n)
        res_dict = manager.dict()
        res_dict.update(self.res_dict) # if self.res_dict is empty then nothing will happen
        
        run_params = []
        for params in self.combined_params:
            
            key = dict(zip(self.original_keys,params))
            key = str(key).replace('{', '').replace('}','')
            if key in res_dict and len(res_dict[key]['r'])==self.cv: # r key always updates first so if that is not updated, the rest arent either
                continue
            #if "'n_gcn': 10" in key and "'n_cnn': 10" in key: # removing big models because RAM issues. Will keep to run later 
            #    res_dict[key] = manager.dict() # keeping params that were not run as an empty dict 
            #    continue # empty dict will not be updated
            else:   
                run_params += [params]
                res_dict[key] = manager.dict()
                res_dict[key]['r'] = manager.list()
                res_dict[key]['r2'] = manager.list()
                res_dict[key]['rmse'] = manager.list()
                res_dict[key]['train_history'] = manager.list()
                res_dict[key]['train_kmers'] = manager.list()
                res_dict[key]['test_kmers'] = manager.list()
                res_dict[key]['train_labels'] = manager.list()
                res_dict[key]['test_labels'] = manager.list()
                res_dict[key]['test_pred'] = manager.list()
                res_dict[key]['train_pred'] = manager.list()

        print('testing {} combinations'.format(len(run_params)), flush=True) 
        po = Pool(len(avail_gpus))
        
         
        r = po.map_async(self.run_cv ,
                     ((kmer_list, pA_list, params, avail_gpus,best_score_params, res_dict, labels) for params in run_params))

        r.wait()
        print(r.get())
        po.close()
        po.join()        

        self.best_score = best_score_params[0]
        self.best_params = best_score_params[1] #unclean dict
        self.clean_up_params()

        new_res_dict = {}
        for params in res_dict.keys():
            new_res_dict[params] = {}

            for key in res_dict[params].keys():
                new_res_dict[params][key] = list(res_dict[params][key])

        self.cv_results=new_res_dict


    def clean_up_params(self):
        
        wanted_keys =  list(self.original_keys)
        new_keys = list(self.best_params.keys())
        unwanted_params = [key for key in new_keys if key not in wanted_keys]
        
        for param in unwanted_params:
            del self.best_params[param]
    
    def save_dict(self,res_dict):
    
        new_res_dict = {}
        for drug in res_dict.keys():
            new_res_dict[drug] = {}

            for key in res_dict[drug].keys():
                new_res_dict[drug][key] = list(res_dict[drug][key])

        
        prp = str(os.environ['PRP'])
        local_out = str(os.environ['MYOUT'])
        s3out = str(os.environ['S3OUT'])

        np.save('.'+local_out+self.res_fn, new_res_dict)

        session = boto3.session.Session(profile_name="default")
        bucket = session.resource("s3", endpoint_url=prp).Bucket("stuartlab")
        bucket.upload_file('.'+local_out+self.res_fn, s3out+self.res_fn)

    def run_cv(self, args):
        
        kmer_list, pA_list, = args[0], args[1]
        params = args[2] # tuple
        avail_gpus = args[3]
        best_score_params = args[4]
        res_dict = args[5]
        labels = args[6] 
        print(labels is None)
        model_params = dict(zip(self.original_keys,params))
 
        print('testing params', model_params, flush=True)
        key =  str(model_params).replace('{', '').replace('}','')
        if labels is None:
            splitter = ShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=42).split(kmer_list) 
        elif labels is not None:
            splitter = StratifiedShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=42).split(kmer_list, labels)
        
        cv_rmse = [] # list of rmses from all folds
        
        for train_idx, test_idx in splitter:
            kmer_train = kmer_list[train_idx]
            pA_train = pA_list[train_idx]

            #kmer_test = kmer_list[test_idx]
            #pA_test = pA_list[test_idx]

            test_n = len(test_idx)
            valid_idx = np.random.choice(test_idx, int(test_n/2),replace=False)
            test_idx = np.array([x for x in test_idx if x not in valid_idx])
            
            kmer_valid = kmer_list[valid_idx]
            pA_valid = pA_list[valid_idx]
    
            kmer_test = kmer_list[test_idx]
            pA_test = pA_list[test_idx]
            
            assert len(kmer_train)+len(kmer_valid)+len(kmer_test) == len(kmer_list)
           
            #print('getting adj', flush=True) 
            # getting adj and feature matrix for smiles
            A_train, X_train = get_AX(kmer_train, n_type = self.n_type)
            gcn_filters_train = initialize_filters(A_train)
            A_test, X_test = get_AX(kmer_test, n_type = self.n_type)
            gcn_filters_test = initialize_filters(A_test)
            A_valid, X_valid = get_AX(kmer_valid, n_type = self.n_type)
            gcn_filters_valid = initialize_filters(A_valid)

            model_params['X'] = X_train
            model_params['filters'] = gcn_filters_train

            model = self.model(**model_params)

            print('running_model', flush=True)
            r,r2,rmse_score, train_hist,test_pred, train_pred = run_params((model,pA_train,pA_test,pA_valid,X_train, gcn_filters_train, X_test, gcn_filters_test,X_valid, gcn_filters_valid, avail_gpus))
           
            print('updating_dict',flush=True) 
            res_dict[key]['r'] += [r]
            res_dict[key]['r2'] += [r2]
            res_dict[key]['rmse'] += [rmse_score]

            res_dict[key]['train_history']  += [train_hist]
            res_dict[key]['train_kmers'] += [kmer_train]
            res_dict[key]['test_kmers'] += [kmer_test]
            res_dict[key]['train_labels'] += [pA_train]
            res_dict[key]['test_labels'] += [pA_test]
            res_dict[key]['test_pred'] += [test_pred]          
            res_dict[key]['train_pred'] += [train_pred]

            cv_rmse += [rmse_score]

        mean_rmse = np.mean(cv_rmse)
        print('mean_rmse', mean_rmse)
        if best_score_params[0] is None or mean_rmse < best_score_params[0]:
            print('updating')
            best_score_params[0] = mean_rmse
            best_score_params[1] = model_params
        
        if self.res_fn is not None:
            self.save_dict(res_dict)
            

    
        
    
        
if __name__ == "__main__":
    # for testing

    print(len(gen_all_kmers()))
    #param_dict = dict(b=[1,2,3], a=[4,5])

    #c = GPUGSCV(param_dict)
     
    #fn = "./ont_models/r9.4_180mv_450bps_6mer_DNA.model"
    #kmer_list, pA_list = kmer_parser(fn)
    #fn = "../ont_models/r9.4_180mv_70bps_5mer_RNA.model"
    #all_kmers, all_pas = kmer_parser(fn)
    #kmer_mat1, pa_list = kmer_parser(fn, exclude_base="G")
    
    #kmer_mat2, pa_list = kmer_parser(fn, exclude_base="C")

    #print(len(set(all_kmers)))
    #print(len(set(kmer_mat2)))

    #print(len(set(kmer_mat1)), len(set(kmer_mat2)))

    #print(len(set(kmer_mat1)|set(kmer_mat2)))
        
    '''
    for test_size,kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat in cv_folds(kmer_mat, pa_list, folds=10):
        print(kmer_train_mat.shape)
        print(kmer_test_mat.shape)
        print(pA_train_mat.shape)
        print(pA_test_mat.shape) 
    all_data, all_pA, all_labels = cg_mg_combine()
    for test_size,kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat in cv_folds(all_data, all_pA, labels=all_labels, folds=5):
        print(kmer_train_mat.shape)
        print(kmer_test_mat.shape)
        print(pA_train_mat.shape)
        print(pA_test_mat.shape)
        
    kmer_list, pA_list = kmer_parser(fn)
    for kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat in cv_folds(kmer_list, pA_list):
        print(kmer_train_mat.shape)
        print(kmer_test_mat.shape)
        print(pA_train_mat.shape)
        print(pA_test_mat.shape)
    

    for _,kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat in base_folds(kmer_list, pA_list):
        print(kmer_train_mat.shape)
        print(kmer_test_mat.shape)
        print(pA_train_mat.shape)
        print(pA_test_mat.shape)

        #print(kmer_train_mat[0])
        #print(kmer_test_mat[0])
    '''
