using NeuralNetwork;
using System;
using System.Collections.Generic;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNet nn = new NeuralNet(2,new int[] { 2,1}, x => 1 / (1 + Math.Exp(-x)), y => y * (1 - y));

            string d = "sL;sW;-12,2643508140491;-12,256593505583;-13,6443434142424;-13,6501479038976;eW;sB;18,4664323920447;6,36497931231579;eB;sL;sW;9,92415557170283;-10,1213229187675;eW;sB;-4,8587400840317;eB;";


            nn.Predict(new double[] { 0, 0 }).Print();
            nn.Predict(new double[] { 0, 1 }).Print();
            nn.Predict(new double[] { 1, 1 }).Print();
            nn.Predict(new double[] { 1, 0 }).Print();

            Console.WriteLine("*********");
            nn.LoadState(d);

            List<Tuple<double[], double[]>> dataSet = new List<Tuple<double[], double[]>>();
            dataSet.Add(new Tuple<double[], double[]>(new double[] { 0, 0 }, new double[] { 0 }));
            dataSet.Add(new Tuple<double[], double[]>(new double[] { 0, 1 }, new double[] { 1 }));
            dataSet.Add(new Tuple<double[], double[]>(new double[] { 1, 0 }, new double[] { 1 }));
            dataSet.Add(new Tuple<double[], double[]>(new double[] { 1, 1 }, new double[] { 0 }));



            //Console.WriteLine("Training");
            //Random r = new Random();
            //for (int i = 0; i < 500000; i++)
            //{
            //    var trainSet = dataSet[r.Next(dataSet.Count)];
            //    nn.Train(trainSet.Item1, trainSet.Item2, 0.2);
            //}
            //Console.WriteLine("Training Complete");

            //Console.WriteLine("");
            //Console.WriteLine("*********");
            //Console.WriteLine("");
            //var d = nn.SaveState();
            //Console.Write(d);
            //Console.WriteLine("");
            //Console.WriteLine("*********");
            nn.Predict(new double[] { 0, 0 }).Print();
            nn.Predict(new double[] { 0, 1 }).Print();
            nn.Predict(new double[] { 1, 1 }).Print();
            nn.Predict(new double[] { 1, 0 }).Print();
            Console.ReadLine();
        }
    }
}
