*.t7 dataset format:
{
  trainData : 
    {
      data : ByteTensor - size: 50000x3x32x32
      labels : ByteTensor - size: 50000
    }
  testData : 
    {
      data : ByteTensor - size: 10000x3x32x32
      labels : ByteTensor - size: 10000
    }
}

labels: 1-numClasses