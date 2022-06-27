"""
Synthetic Discovery and Design (SD2)
Perovskite Design of Experiments

Utilize Gaussian Process with bayesian optimization
to select optimal experiments based on uncertainty
criteria and maximization of knowledge gained.

Requires python 3.5 for batch values greater than 1

example run
python perovskite_doe.py -T training.csv -t stateset.csv -i 0057_phase1 -a XFYICZOIWSBQSK-UHFFFAOYSA-N
                         -f _rxn_M_inorganic,_rxn_M_organic,_rxn_M_acid -c 16
"""

import sys, time
import csv
import argparse
from scipy.spatial import ConvexHull, distance
import pandas as pd
import numpy as np
from sklearn import preprocessing
import GPy
import cma
import GPyOpt as gp
import matplotlib.pyplot as plt
import scipy.stats as st


def arguments(args=None):
    parser = argparse.ArgumentParser(description='DoE Perovskites')

    parser.add_argument('-T', '--train_file',   dest='train_file',
                        help='Training data (csv)')
    parser.add_argument('-t', '--test_file',    dest='test_file',
                        help='Test data (csv)')
    parser.add_argument('-i', '--identifier',   dest='identifier')
    parser.add_argument('-a', '--amine',        dest='amine',
                        help='Amine InChiKey to focus optimization on.', 
                        required=False)
    parser.add_argument('-j', '--jitter',       dest='jitter',
                        help='Tunable exploratory parameter, increase to explore',
                        required=False, default=0.02)
    parser.add_argument('-f', '--features',     dest='features', required=False,
                        help='Add features to train delimited by ,',
                        default=False)
    parser.add_argument('-v', '--verbose',      dest='verbose', required=False,
                        help='Verbose output', action='store_false',
                        default=True)
    parser.add_argument('-c', '--cores',        dest='cores',   required=False,
                        help='Cores available for process',
                        default=10)
    parser.add_argument('-b', '--batch',        dest='batch',   required=False,
                        help='Number of samples to select for next experiment',
                        default=1)
    return parser.parse_args()


def main(args=None):
    train_file = args.train_file
    test_file = args.test_file
    features = args.features
    features = features.split(',')
    jitter = float(args.jitter)
    verbose = args.verbose
    cores = args.cores
    identifier = args.identifier
    amine = args.amine
    batch_size = args.batch
    outfile = open(identifier + "_singleton_results.out", 'w')

    test_df = pd.read_csv(test_file, engine='python')
    train_df = pd.read_csv(train_file, engine='python')

    constraints = hull_inequalities(test_df[features])

    train_df = prepare_training_data(train_df, features, constraints, amine)

    domain = create_domain(test_df, features)

    if verbose:
         print('INFO: Training dataframe shape (%s, %s)' % train_df.shape)
         if amine:
             print('INFO: Amine ' + amine)
         print('INFO: Applying %s constraints' % len(constraints))

    predicted_experiments, model = gpy_predictions(train_df, domain, constraints,
                                                   cores, identifier, jitter,
                                                   batch_size, test_df,features)
    vial_ids = get_vial_ids_from_predicted_experiments(features, outfile, predicted_experiments,
                                                       test_df[features],model)
    handle = submission_template(vial_ids, identifier, features)
    createfigures(identifier, test_df[features].values, model, features)
    if verbose: print('INFO: Created submission template %s' % handle)
    return 0

def createfigures(identifier, data, model, features):
    """
    Creates a 2d representation of the statespace with predicted crystal scores in black and uncertainty in blue
    Mostly used to determine if the model improves during active learning phase
    """
    outdata = open(identifier + "stateset_predictions.txt",'w')
    outdata.write("predicted_score\tprobability_u3")
    for val in features:
        outdata.write("\t" + val)
    outdata.write("\n")
    ydata = []
    xdata = []
    hdata = []
    ldata = []
    x = 0
    for conc in data:
        y_pred = model.model.predict(conc)
        y_val = str(y_pred[0][0]).replace("[","")
        y_val = y_val.replace("]","")
        y_val = y_val.replace("-","")
        y = float(y_val)
        stdev = str(y_pred[1][0]).replace("[","")
        stdev = stdev.replace("]","")
        s = float(stdev)
        h = y + s
        l = y - s
        xdata.append(x)
        x += 1
        z = (3.0-y)/s
        p = 1 - st.norm.cdf(z)
        ydata.append(y)
        hdata.append(h)
        ldata.append(l)
        outdata.write(str(y) + "\t" + str(p))
        for val in conc:
            outdata.write("\t" + str(val))
        outdata.write("\n")

    outdata.close()
    plt.scatter(xdata,hdata, color = 'blue')
    plt.scatter(xdata,ldata, color = 'blue')
    plt.scatter(xdata,ydata, color = 'black')
    plt.ylabel("Predicted Crystal Score")
    plt.xlabel("Stateset Position")
    plt.savefig(identifier + "_plot.png")


def prepare_training_data(train_df=None, features=None, constraints=None,
                          amine=None):
    """
    Subset the training dataframe by the features provided on command line along
    with the feature we want to train on, crystalscore.

    :param train_df: pd.dataframe, contains the entire training data set
    :param features: list, contains features that will be used for training the
                     GP model
    :param constraints: list, containing dicts of 2D plane inequalities used to
                        constrain the optimization domain futher along with
                        region to select next locations
    :param amine: str, InChiKey of the amine to subset the training df on
    :return: pd.dataframe, a dataframe with the X and Y unscaled data
    """
    if amine:
        train_df = train_df[train_df['_rxn_organic-inchikey'] == amine]
    constrained_rows = apply_constraint(train_df[features].values, constraints)
    train_ = train_df[features].multiply(constrained_rows.flatten(), axis=0)
    train_ = train_df[(train_.T != 0).any()]

    if not features:
        raise Exception('ERROR: Found no features to subset training on, ' +
                        'exiting.\n' +
                        'Use -f to provide list of features to train on')
    train_ = train_[features + ['_out_crystalscore']]
    return train_


def apply_constraint(x=None, constraints=None):
    """
    Borrowed constraint function from GPyOpt that we will use to decrease the
    points tested inside the domain of the given State Set. Having information
    about points sampled outside the State Set does us no good and will simply
    increase the compute time (remember, n^3).

    :param x: pd.DataFrame, the training dataframe
    :param constraints: list, containing 2D plane inequalities used to
                        constrain the optimization domain futher along with
                        region to select next locations
    :return: numpy Series, same length vector of 0 or 1 denoting if point falls
             within or out of State Set domain
    """
    x = np.atleast_2d(x)
    I_x = np.ones((x.shape[0],1))
    if constraints is not None:
        for d in constraints:
            try:
                constraint = eval('lambda x: ' + d['constrain'])
                ind_x = (constraint(x)<0)*1
                I_x *= ind_x.reshape(x.shape[0],1)
            except:
                print('failed')
                raise
    return I_x


def get_vial_ids_from_predicted_experiments(features, outfile, predicted_experiments=None, test_df=None,model=None):
    """
    Use a nearest-neighbor approach to locate a vial ID from the test set that
    is most similar to our set of predicted experiments.

    :param predicted_experiments: np.array, 2D containing predicted sample
                                  reaction conditions
    :param test_df: pandas.dataframe, with test set vial ids and reaction
                    conditions
    :return: list, of vial ids from the test set
    """
    test = 1
    outfile.write("test\tclosest_test\ty_val\tstdev")
    for val in features:
    	outfile.write("\t" + val)
    outfile.write("\n")
    test_df = test_df.values
    vial_ids = []
    for sample in predicted_experiments:
        closest_test, closest_params = closest_node(sample, test_df)
        print (closest_test)
        print(closest_params)
        y_pred = model.model.predict(closest_params)
        y_val = str(y_pred[0][0]).replace("[","")
        y_val = y_val.replace("]","")
        y_val = y_val.replace("-","")
        stdev = str(y_pred[1][0]).replace("[","")
        stdev = stdev.replace("]","")
        outfile.write(str(test) + "\t" + str(closest_test) + "\t" + y_val + "\t" + stdev)
        for val in closest_params:
        	outfile.write("\t" + str(val))
        outfile.write("\n")
        test+=1
        vial_ids.append(closest_test)

    outfile.close()

    return vial_ids


def closest_node(node, nodes):
    """
    Uses scipy.spatial.distance.cdist to get the minimum distance node out of a
    list of nodes. In our case we use it to find the nearest Test Set vial ID to
    the predicted reaction conditions we want to test.

    :param node: list, containing a single experiments worth of reaction
                 conditions
    :param nodes: np.array, 2D containing all the Test Set reaction conditions
    :return: int, the index == the vial ID nearest to the query node and
             list, containing the reaction conditions for the nearest vial ID
    """
    closest_index = distance.cdist([node], nodes).argmin()
    return closest_index, nodes[closest_index]


def submission_template(vial_ids=None, identifier=None, features=None):
    """
    Write the next experiments to a submission template following Haverfords
    format. Use an identifier in the name handle to identify what experiment the
    predictions are for, such as an amine.

    :param vial_ids: list, containing the vial IDs to be evaluated in the
                     experiment
    :param identifier: string, identifies which experiment the suggested
                       reactions should be applied to
    :param features: list, of strings containing column names to use as features
    :return: string, file name handle which holds the predictions
    """
    date = time.strftime('%Y%m%d')
    handle = '_'.join([date, 'Foundry', identifier])
    handle += '.csv'
    columns = ['RunID_Vial', 'out_predicted', 'modelname', 'participantname',
               'notes']
    #columns = ['dataset', 'name', '_rxn_M_inorganic', '_rxn_M_organic',
    #           'predicted_out', 'score']
    modelname = 'BO-GP'
    user = 'AC-Foundry'
    note = identifier + ';' + ';'.join(features)
    output_rows = []
    print ("submission")

    for i in range(len(vial_ids)):
        output_rows.append([vial_ids[i], 4, modelname, user, note])
    holder_df = pd.DataFrame(output_rows, columns=columns)
    holder_df.to_csv('./' + handle, index=False)
    return handle


def gpy_predictions(dataframe=None, domain=None, constraints=None, cores=1,
                    identifier=None, jitter=0.02, batch_size=48, test_df=None, features = None):
    """
    Optimization routine from GPyOpt. Will train a gaussian process on the
    dataframe within the domain and following any constraints. GP evaluation
    occurs during suggestion of next locations call, and in series, which can be
    distributed to each core by using the local penalization batch selection
    acquisition criteria. Will always train for maximum '_out_crystalscore'.

    :param dataframe: pandas.dataframe, containing X and Y values, unscaled
    :param domain: dictionary, in which the GP will be optimized, defined by
                   min and max of each dimension
    :param constraints: dictionary, containing 2D plane inequalities used to
                        constrain the optimization domain futher along with
                        region to select next locations
    :param identifier: string, identifies which experiment the suggested
                       reactions should be applied to
    :param cores: int, number of cores available for compute
    :param jitter: float, exploratory paramter for the acquisition function
    :param batch_size: int, number of samples to suggest for next experiment
    :return: np.array, of 2D containing conditions for the next round of
             experiments
    """
    X = dataframe.drop('_out_crystalscore', axis=1) 
    X = dataframe.drop('_out_crystalscore', axis=1).values
    Y = dataframe['_out_crystalscore'].values
    print("Look here")
    print(X)
    print(Y)


    acq = 'EI'
    jitter = jitter
    batch_size = batch_size
    evaluator = 'local_penalization'
    kernel = GPy.kern.Matern32(len(domain), ARD=True)
    gp_model = gp.methods.BayesianOptimization(f=None,
                                               kernel = kernel,
                                               batch_size=batch_size,
                                               domain=domain,
                                               constraints=constraints,
                                               evaluator_type=evaluator,
                                               acquisition_type=acq,
                                               acquisition_jitter=jitter,
                                               X=X,
                                               Y=-1*Y.reshape(-1, 1),
                                               normalize_Y=False,
                                               exact_feval=False,
                                               ARD=True,
                                               model_type='GP',
                                               de_duplication=True,
                                               num_cores=cores
                                               #maximize=True
                                               )

    predicted_experiments = gp_model.suggest_next_locations()
    y_pred = gp_model.model.predict(predicted_experiments)
    ylist = ','.join(str(s) for s in y_pred)

    """
    outfile = open("0037_yvals.txt",'w')
    for x in range(len(Y)):
    	#print(str(Y[x]) + "\t" + str(ylist[x]) + "\n") 
    	outfile.write(str(Y[x]) + "\n")
    with open('0037_ytest.csv','w') as out:
    	csv_out=csv.writer(out)
    	csv_out.writerow(['name','num'])
    	for row in y_pred:
    		csv_out.writerow(row)
    outfile.close()
    """

    gpanalyze(gp_model,domain,X, test_df[features])
    return predicted_experiments, gp_model


def hull_inequalities(dataframe=None, reduction=False):
    """
    Calculate 2D planes of a convex hull surrounding the test set points.
    Reduce the number of planes based on a similarity metric defined as
    requiring each variable in an inequality to be within a value from another
    inequality.
    eg. 0.2x + 1.2y + 3 <= 0 will be compared against 0.21x + 1.2y + 3 <= 0 and
        be marked for removal since all values are within 0.01 from each other;
        forming redundant planes.

    :param dataframe: pandas.dataframe of X values of the Test set
    :param reduction: bool, reduce simplices of hull based on similarity to one
                      another
    :return: dictionary with names and equations for constraints to be passed
             to GPyOpt during optimization and experiment selection
    """
    hull = ConvexHull(dataframe.values)
    equations = list([tuple(x) for x in np.array(hull.equations)])
    
    if reduction:
        removals = set()
        for i, eq in enumerate(equations):
            eq = list(eq)
            compare = equations[:]
            compare.pop(i)
            for z in compare:
                z = list(z)
                check_diff = [abs(eq[j] - z[j]) for j, x in enumerate(z)]
                if all(x <= 0.01 for x in check_diff):
                    if not removals:
                        removals.add(i)
                    for r in removals:
                        to_remove = None
                        if all(x <= 0.01 for x in [abs(equations[r][j] - z[j]) for j, x in enumerate(z)]):
                            continue
                        else:
                            to_remove = i
                    if to_remove:
                        removals.add(to_remove)

        reduced_equations = [x for i, x in enumerate(equations) if i not in removals]
        equations = reduced_equations

    constraints = []
    for i, eq in enumerate(equations):
        equation = '%s * x[:,0] + %s * x[:,1] + %s * x[:,2] + %s' % (eq)
        cons = {'name': 'const_%s' % str(i + 1),
                'constrain': equation,
                'constraint': equation}
        constraints.append(cons)
    return constraints


def create_domain(dataframe=None, features=None):
    """
    Create a domain for the GPyOpt BayesianOptimization method. This contains an
    entry for each feature we are training the model on, defined by a minimum 
    and maximum value contained in the dataframe column for that term. This only
    handles continuous features for now.

    :param dataframe: pd.dataframe, contains the training data
    :param features: list, of strings containing column names to use as features
    :return: list, of dictionaries, one for each feature
    """
    domain = []
    for z in features:
        domain.append({'name': str(z),
                       'type': 'continuous',
                       'domain': [round(dataframe[z].min(), 4),
                                  round(dataframe[z].max(), 4)]})

    #print("domain")
    #print(domain)
    return domain
def gpanalyze(model,domain, X, test_df):
    """
    Identifies local minima within the stateset to identify points that have the highest likelyhood of being crystals
    """
    minval = 100
    maxval = 100

    mins = []
    maxs = []
    for _ in domain:
        d = _['domain']
        if len(d) > 1:
        	mins.append(d[0])
        	maxs.append(d[1])
        	if d[1] < maxval:
        		minval = d[0]
        		maxval = d[1]

    # This function will wraps the GP.predict() for compatibility with CMA-ES, mainly by 
    # reshaping the vector
    def func(vec):
        tvec = np.concatenate([vec,[0.0]* len(vec)])
        tvec = tvec.reshape(1,2*len(vec))
        ans,var = model.model.predict(tvec)
        return ans[0][0]


    # Optimization parameters for CMA-ES
    # Sigma should be about ~%20-25 of your domain size
    options = {}
    options['tolfun'] = 1e-10
    options['bounds'] = mins , maxs
    sigma = (maxval - minval) *0.1

    cmaesout = open("cma_es_guesses.out",'w')
    test_df = test_df.values

    for i in range(len(X)):
        vial_ids =[]
        frac = i+1/7
        initguess = X[i]

        a = cma.fmin(func, initguess, sigma, options=options)

        answer   = a[0]
        best_y   = a[1]
        cmaesobj = a[-2]
        best = cmaesobj.gp.pheno(cmaesobj.mean, into_bounds=cmaesobj.boundary_handler.repair)
        closest_test, closest_params = closest_node(answer,test_df)
        
        print ("closest_test " + str(closest_test))
        print("closest_params " + str(closest_params))
        y_pred = model.model.predict(closest_params)
        y_val = str(y_pred[0][0]).replace("[","")
        y_val = y_val.replace("]","")
        y_val = y_val.replace("-","")
        stdev = str(y_pred[1][0]).replace("[","")
        stdev = stdev.replace("]","")
        quickout.write("best_y\t" + str(best_y) + "\t" + str(closest_test) + "\t" + str(closest_params) + "\t" + y_val + "\t" + stdev + "\n")

    cmaesout.close()

if __name__== "__main__":
    args = arguments(sys.argv[1:])
    status = main(args)
    sys.exit(status)
