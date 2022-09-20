import pandas as pd
from utils.all_utils import preparedata,save_plot
from utils.model import Perceptron

def main(eta,epoch,data,modelName,plotName):
    OR = pd.DataFrame(data)

    X, y = preparedata(OR)
    ETA = 0.1
    EPOCH = 10
    model_or = Perceptron(eta=ETA, epochs=EPOCH)
    model_or.fit(X, y)

    model_or.save(filename=modelName, model_dir='model')

    save_plot(OR, filename=plotName, model=model_or)



if __name__=='__main__':
    OR = {
        'x1': [0, 0, 1, 1],
        'x2': [0, 1, 0, 1],
        'Y': [0, 1, 1, 1]
    }
    ETA=0.1
    EPOCH=10

    main(eta=ETA,epoch=EPOCH,data=OR,modelName='or.model',plotName='or.png')



