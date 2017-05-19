package jp.eiya.dl4j.step1

import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.api.*
import org.deeplearning4j.nn.multilayer.*
import org.deeplearning4j.nn.weights.*

import org.deeplearning4j.datasets.datavec.*
import org.deeplearning4j.eval.*

import org.deeplearning4j.optimize.listeners.*

import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit

import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
 * 線形データ分類
 * @see https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierLinear.java
 */
class MLPClassifierLinear {
  final static layer = [
     input  : 2,
     hidden : 20,
     output : 2
  ]

  static class DenseLayerBuilder extends DenseLayer.Builder{} 
  static class OutputLayerBuilder extends OutputLayer.Builder{
    OutputLayerBuilder(org.nd4j.linalg.lossfunctions.ILossFunction lossFunction) {super(lossFunction)}
    OutputLayerBuilder(org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction lossFunction) {super(lossFunction)}
  } 

  final config
  MLPClassifierLinear(){
    config = new NeuralNetConfiguration.Builder().
                 seed(123).
                 iterations(2).
                 optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
                 learningRate(0.01d).
                 updater(Updater.NESTEROVS).
                 momentum(0.9d).
                 list().
                 layer(0,
                   new DenseLayerBuilder().
                       nIn(layer.input).
                       nOut(layer.hidden).
                       weightInit(WeightInit.XAVIER).
                       activation(Activation.RELU).
                       build()
                 ).
                 layer(1,
                   new OutputLayerBuilder(LossFunction.NEGATIVELOGLIKELIHOOD).
                       weightInit(WeightInit.XAVIER).
                       activation(Activation.SOFTMAX).
                       weightInit(WeightInit.XAVIER).
                       nIn(layer.hidden).
                       nOut(layer.output).
                       build()
                 ).
                 pretrain(false).
                 backprop(true).
                 build()
  }

  static createDataSet(String fileName,batchSize) {
    new RecordReaderDataSetIterator({
      def rr = new CSVRecordReader()
      rr.initialize(new FileSplit(new File(fileName)))
      rr
    }(),batchSize,0,2)
  }

  def train(example,int epochNum) {
    def model = new MultiLayerNetwork(config)
    model.init()
    model.setListeners(new ScoreIterationListener(10))
    
    epochNum.times {
      model.fit(example)
    }
    model
  }

  def evalModel(MultiLayerNetwork trainedModel,unknowns) {
    def eval = new Evaluation(layer.output)
    while(unknowns.hasNext()){
      def unknown = unknowns.next()
      eval.eval(
        unknown.labels,
        trainedModel.output(unknown.featureMatrix,false)
      )
    }
    eval
  }
}
