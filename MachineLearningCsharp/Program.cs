using Common.Contract;
using FtSrvCommon.Logs;
using MachineLearningCsharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningTest
{
    class Program
    {
        static void Main(string[] args)
        {
            //Console.WriteLine("Type enter to start");
            //Console.Read();

            Logger.LogLevel = LogLevel.Error;

            int userId = 0;
            int nbDiners = 2;

            DateTime beforePythonInit = DateTime.Now;
            Pythonet_2prod pythonWrapper = new Pythonet_2prod();
            pythonWrapper.InitUser(userId, DateTime.Now, nbDiners);

            DateTime afterPythonInit = DateTime.Now;
            Console.WriteLine("Python initialized in " + (int)(afterPythonInit.Subtract(beforePythonInit).TotalMilliseconds) + " ms");

            //Select product group:
            Event evtProductGroup1Selected = new Event();
            evtProductGroup1Selected.Type = EventType.ProductGroup;
            evtProductGroup1Selected.Parameter = "2";
            pythonWrapper.AddEvent(evtProductGroup1Selected);

            DateTime beforeEvents = DateTime.Now;

            for (int i = 0; i < 30; i++)
            {
                if (i % 20 == 0)
                {
                    Console.WriteLine("i = " + i.ToString());
                }

                //Select product group:
                evtProductGroup1Selected = new Event();
                evtProductGroup1Selected.Type = EventType.ProductGroup;
                evtProductGroup1Selected.Parameter = "2";
                pythonWrapper.AddEvent(evtProductGroup1Selected);

                //Select product:
                Event evtProduct1Selected = new Event();
                evtProduct1Selected.Type = EventType.Product;
                evtProduct1Selected.Parameter = "10";
                pythonWrapper.AddEvent(evtProduct1Selected);

                // Add product to cart
                Event evtProduct1AddedToCart = new Event();
                evtProduct1AddedToCart.Type = EventType.ProductAddedToCart;
                evtProduct1AddedToCart.Parameter = "10";
                pythonWrapper.AddEvent(evtProduct1AddedToCart);
            }
            DateTime afterEvents = DateTime.Now;
            TimeSpan deltaTEvents = afterEvents.Subtract(beforeEvents);

            Console.WriteLine("Added events in " + (int)deltaTEvents.TotalMilliseconds + " ms");

            DateTime beforePredictions = DateTime.Now;
            Dictionary<int, double> productScores = null;
            for (int j = 0; j< 100; j++)
            {
                productScores = pythonWrapper.GetPredictions();
            }
            DateTime afterPredictions = DateTime.Now;

            Console.WriteLine("Received predictions in " + (int)(afterPredictions.Subtract(beforePredictions) .TotalMilliseconds) + " ms");

            Console.WriteLine("Received predictions:");
            if (productScores != null)
            {
                foreach(KeyValuePair<int, double> kvp in productScores)
                {
                    Console.WriteLine(" -> " + kvp.Key.ToString() + "\t" + kvp.Value);
                }
            }

            Console.WriteLine("Type enter to stop");
            Console.Read();
        }
    }
}
