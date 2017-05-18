package jp.eiya.dl4j.step1

import org.nd4j.linalg.factory.Nd4j

class Plotter {

  static window = [
     axis : ['x','y'],
     size : 100,
     min : [x: 0.0d , y:-0.2d ],
     max : [x: 1.0d , y: 0.8d ]
  ]

  static field = window.with {
    def dmap = {ax-> (max[ax]-min[ax])/(size-1) }
    def cnt = 0
    def points = new double[size**2][2]
    size.times {x->
      x = x * dmap('x') + min.x
      size.times {y->
        y = y * dmap('y') + min.y
        points[cnt][0] = x
        points[cnt][1] = y
        cnt++
      }
    }
    Nd4j.create(points)
  }

  def model 
  Plotter(model){
     this.model = model
  }

  def plot(trainDataSet,testDataSet){
     def predictField = model.output(field)
     trainDataSet.with {
       PlotUtil.plotTrainingData(features,labels,field,predictField,window.size)
     }
     testDataSet.with {
       def testField = model.output(features)
       PlotUtil.plotTestData(features,labels,testField,field,predictField,window.size)
     }
  }
}
