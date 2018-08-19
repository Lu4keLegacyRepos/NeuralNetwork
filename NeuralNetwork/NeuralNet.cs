using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralNet
    {
        public List<Layer> _layers { get; set; }


        public NeuralNet(int numOfInputs,int[] layers, Func<double, double> activationFc, Func<double, double> activationFc_derivitiv)
        {
            _layers = new List<Layer>();

            _layers.Add(new Layer(layers[0], numOfInputs) { ActivationFc=activationFc,ActivationFc_derivitiv=activationFc_derivitiv});

            for (int i = 1; i < layers.Length; i++)
            {
                _layers.Add(new Layer(layers[i], _layers[i-1].NeuronsCount) { ActivationFc = activationFc, ActivationFc_derivitiv = activationFc_derivitiv });
                _layers[i - 1].ConnectTo(_layers[i]);
            }

        }


        public Matrix Predict(double[] inp)
        {
            Matrix inputs = new Matrix(inp);
            _layers.First().Inputs = inputs;
            _layers.First().ComputeOut();
            return _layers.Last().Outputs;

        }

        public void Train(double[] inp, double[] outp, double learningRate)
        {
            Matrix inputs = new Matrix(inp);
            Matrix targets = new Matrix(outp);

            _layers.First().Inputs = inputs;
            _layers.First().ComputeOut();

            _layers.Last().UpdateWeights(targets, learningRate);


        }
    }
}
