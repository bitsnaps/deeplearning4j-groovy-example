package jp.eiya.dl4j.step1

import static jp.eiya.dl4j.step1.MLPClassifierLinear.createDataSet as ds

def dataDir = './data/step1'
def trainFile = dataDir+'/train_0101.csv'
def evalFile = dataDir+'/eval_0101.csv'

def classifier = new MLPClassifierLinear()

def model = classifier.train( ds(trainFile,50), 30 )
def eval = classifier.evalModel( model, ds(evalFile,50) )

println eval.stats()

new Plotter(model).plot(
  ds(trainFile,1000).next(),
  ds( evalFile, 500).next()
)

