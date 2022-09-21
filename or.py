import os.path

import pandas as pd
from utils.all_utils import preparedata,save_plot
from utils.model import Perceptron
import logging


log_dir='log'
os.makedirs(log_dir,exist_ok=True)


logging.basicConfig(level=logging.INFO,filename=os.path.join(log_dir,'running_logs.log'),
                    format='[%(asctime)s:%(levelname)s:%(module)s] : %(message)s',      #printing part
                    filemode='a'                                                       #append
                    )



def main(eta,epoch,data,modelName,plotName):
    OR = pd.DataFrame(data)
    logging.info(f'This is raw dataset: \n {OR}')


    X, y = preparedata(OR)
    ETA = 0.1
    EPOCH = 10
    model_or = Perceptron(eta=ETA, epochs=EPOCH)
    model_or.fit(X, y)

    model_or.save(filename=modelName, model_dir='model')

    save_plot(OR, filename=plotName, model=model_or)



if __name__=='__main__':
    gate='OR_gate'
    OR = {
        'x1': [0, 0, 1, 1],
        'x2': [0, 1, 0, 1],
        'Y': [0, 1, 1, 1]
    }
    ETA=0.1
    EPOCH=10

    try:
        logging.info(f'>>>>>>>>>>>>>>>>>>Starting_training for {gate}>>>>>>>>>>>>>>\n')
        main(eta=ETA,epoch=EPOCH,data=OR,modelName='or.model',plotName='or.png')
        logging.info(f"<<<<<<<<<<<<<<<<<<End of Training for {gate}<<<<<<<<<<<<<<<<<<<<<<\n")

    except Exception as e:
        logging.exception(e)
        raise e
