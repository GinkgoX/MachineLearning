from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus

if __name__ == '__main__':
    with open('info.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lense_target = []
    for each in lenses:
        lense_target.append(each[-1])
    #print(lense_target)     #to get the target label
    
    lenseLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenseList = []
    lenseDict = {}
    for label in lenseLabels:
        for each in lenses:
            lenseList.append(each[lenseLabels.index(label)])
        lenseDict[label] = lenseList
        lenseList = []
    #print(lenseDict)    #to create lenseDict
    #print('###################################')
    
    lenseDF = pd.DataFrame(lenseDict)
    #print(lenseDF)    #to create pandas dataframe
    
    encoder = LabelEncoder()
    for col in lenseDF.columns:
        lenseDF[col] = encoder.fit_transform(lenseDF[col])
    #print(lenseDF)    #to convert 0 ~ num_class-1 integer
    
    model = tree.DecisionTreeClassifier(max_depth = 4)
    model = model.fit(lenseDF.values.tolist(), lense_target)
    dotData = StringIO()
    tree.export_graphviz(model, out_file = dotData, feature_names = lenseDF.keys(),
                        class_names = model.classes_, filled = True, rounded = True,
                        special_characters = True)
    graph = pydotplus.graph_from_dot_data(dotData.getvalue())
    graph.write_pdf('lenses.pdf')
    #age : pre, presbyopic, young
    #prescript : myope, hyper
    #astigmatic : no, yes
    #treaRate : normal, reduced
    #2           1          1         0 hard
    #0           0          0         1 no lense
    #0           0          0         0 soft
    '''
    young	hyper	yes	normal	hard
    pre	myope	no	reduced	no lenses
    pre	myope	no	normal	soft
    '''
    print(model.predict([[2, 1, 1, 0]]))
    print(model.predict([[0, 0, 0, 1]]))
    print(model.predict([[0, 0, 0, 0]]))


