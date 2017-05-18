package jp.eiya.dl4j.step1

import static jp.eiya.dl4j.step1.Sample0101.createDataSet as ds

def sample = new Sample0101()

def model = sample.train(ds('./train_0101.csv',50),30)
def eval = sample.evalModel(model,ds('./eval_0101.csv',50))

println eval.stats()

new Plotter(model).plot(
  ds('./train_0101.csv',1000).next(),
  ds('./eval_0101.csv',500).next()
)

