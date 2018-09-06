using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralNet
    {
        internal List<Layer> NeuralLayers { get; set; }


        public NeuralNet(int numOfInputs,int[] layers, Func<double, double> activationFc, Func<double, double> activationFc_derivitiv)
        {
            NeuralLayers = new List<Layer>();

            NeuralLayers.Add(new Layer(layers[0], numOfInputs) { ActivationFc=activationFc,ActivationFc_derivitiv=activationFc_derivitiv});

            for (int i = 1; i < layers.Length; i++)
            {
                NeuralLayers.Add(new Layer(layers[i], NeuralLayers[i-1].NeuronsCount) { ActivationFc = activationFc, ActivationFc_derivitiv = activationFc_derivitiv });
                NeuralLayers[i - 1].ConnectTo(NeuralLayers[i]);
            }

        }

        public string SaveState()
        {
            string data = "";
            foreach (var l in NeuralLayers)
            {
                data += "sL;";
                data += "sW;";
                var dta = l.Weights.ToArray();
                foreach (var val in dta)
                {
                    data += val.ToString() + ";";
                }
                data += "eW;";

                dta = l.Bias.ToArray();
                data += "sB;";
                foreach (var val in dta)
                {
                    data += val.ToString() + ";";
                }

                data += "eB;";
            }
            return data;
        }

        public void LoadState(string state)
        {
            string[] data = state.Split(';');

            int layerIndex = -1;
            Layer actualLayer = null;
            List<double> loadedData = new List<double>();
            foreach (var txt in data)
            {
                if (txt.Contains("sL"))
                {
                    layerIndex++;
                    actualLayer = NeuralLayers[layerIndex];
                    continue;
                }
                if (txt.Contains("sW") || txt.Contains("sB"))
                {
                    loadedData.Clear();
                    continue;
                }
                if (txt.Contains("eW"))
                {
                    actualLayer.Weights.Data = loadedData.ToArray();
                    continue;
                }
                if (txt.Contains("eB"))
                {
                    actualLayer.Bias.Data = loadedData.ToArray();
                    continue;
                }
                if (string.IsNullOrEmpty(txt)) continue;
                loadedData.Add(double.Parse(txt));

            }
                
        }
        public Matrix Predict(double[] inp)
        {
            Matrix inputs = new Matrix(inp);
            NeuralLayers.First().Inputs = inputs;
            NeuralLayers.First().ComputeOut();
            return NeuralLayers.Last().Outputs;

        }

        public void Train(double[] inp, double[] outp, double learningRate)
        {
            Matrix inputs = new Matrix(inp);
            Matrix targets = new Matrix(outp);

            NeuralLayers.First().Inputs = inputs;
            NeuralLayers.First().ComputeOut();

            NeuralLayers.Last().UpdateWeights(targets, learningRate);


        }
    }
}
