using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Python.Runtime;
using System.Diagnostics;
using System.Threading;
using Common.Contract;
using Common.Log;
using FtSrvCommon.Logs;
using DbModel;
//using System.Data.Common.Contract;

namespace MachineLearningCsharp
{
    public class Pythonet_2prod
    {
        /// <summary>
        /// The logger for this class.
        /// </summary>
        private static ILog logger = Logger.Instance(AppArea.ShopSvc, typeof(Pythonet_2prod));

        public static PredictionMode UsedPredictionMode = Pythonet_2prod.PredictionMode.PredictionWithProductPriceOptimization;

        /// <summary>
        /// Stores instance of Pythonet_2prod class per user.
        /// Key = User ID
        /// Value = Tuple (Item1 = Date and Time of insertion, Item2 = Pythonet_2prod wrapper for user)
        /// </summary>
        private static Dictionary<int, Tuple<DateTime, Pythonet_2prod>> instancePerUser = new Dictionary<int, Tuple<DateTime, Pythonet_2prod>>();

        public enum PredictionMode
        {
            PredictionWithProductScoreOptimization,
            PredictionWithProductPriceOptimization
        }
        /// <summary>
        /// Semaphore to prevent multithreaded accesses to the Python engine initialization.
        /// </summary>
        private static object pythonInitializationSemaphore = new object();

        public static bool PythonInitialized
        {
            get { return PythonEngine.IsInitialized; }
        }

        // Instance parameters:
        private int userId = 0;
        private DateTime timestamp;
        private int nbDiners = 1;

        /// <summary>
        /// Reference to our custom python module.
        /// </summary>
        private dynamic pythonModule = null;

        /// <summary>
        /// Reference to python method used to retrieve a feature out of a raw variable.
        /// In ML, a feature is an other representation of  data so that it can be used by an algorithm.
        /// </summary>
        private dynamic getFeatures = null;

        private dynamic userProfile = null;

        public static void InitPython()
        {
            try
            {
                if (PythonEngine.IsInitialized)
                {
                    return;
                }

                lock (pythonInitializationSemaphore)
                {
                    if (PythonEngine.IsInitialized)
                    {
                        return;
                    }

                    logger.Debug(nameof(InitPython), "Initializing python engine.");

                    string anacondaHome = null;
                    
                    if (System.Configuration.ConfigurationManager.AppSettings.AllKeys.Contains("AnacondaHome"))
                    {
                        anacondaHome = System.Configuration.ConfigurationManager.AppSettings["AnacondaHome"];
                    }

                    if (string.IsNullOrWhiteSpace(anacondaHome))
                    {
                        // Default value:
                        anacondaHome = "C:\\Program Files\\Anaconda";
                    }

                    // defining a path to execute the right python scrip for the customer in question ( ex:pizzup)
                    // see App.config line 12: <add key="PythonContentPath" value="C:\Users\Pham Antoine\Documents\Dev\DevCuddlUpSolution\server\MachineLearning\MachineLearningCsharp\bin\x64\Debug\PythonContent\Pizzup" />
                    string pythonContentPath = null;
                    if (System.Configuration.ConfigurationManager.AppSettings.AllKeys.Contains("PythonContentPath"))
                    {
                        pythonContentPath = System.Configuration.ConfigurationManager.AppSettings["PythonContentPath"];
                    }
                    else
                    {
                        pythonContentPath = @"C:\dev\DevCuddlUpSolution\server\MachineLearning\MachineLearningCsharp\bin\x64\Debug\PythonContent";
                    }

                    string pythonPath = $"{ anacondaHome}\\Lib;"
                         + $"{anacondaHome}\\Lib\\site-packages;"
                         + $"{anacondaHome}\\Library\\bin;"
                         + $"{pythonContentPath};"
                         + $"{pythonContentPath}\\params;"
                         + $"{pythonContentPath}\\prediction;"
                         + $"{pythonContentPath}\\training;"
                        ;

                    string newPath = $"{anacondaHome}\\Library\\bin;{anacondaHome}\\Library;{anacondaHome};";
                    newPath += Environment.GetEnvironmentVariable("PATH");

                    Environment.SetEnvironmentVariable("PythonPath", pythonPath, EnvironmentVariableTarget.Process);
                    Environment.SetEnvironmentVariable("PythonHome", anacondaHome, EnvironmentVariableTarget.Process);
                    Environment.SetEnvironmentVariable("PATH", newPath, EnvironmentVariableTarget.Process);

                    try
                    {
                        PythonEngine.Initialize();
                    }
                    catch (AccessViolationException)
                    {
                        // CAUTION: We have some times: 
                        // System.AccessViolationException : Attempted to read or write protected memory. This is often an indication that other memory is corrupt.
                        // So try again...
                        // Retry Nb. 1:
                        try
                        {
                            PythonEngine.Initialize();
                        }
                        catch (AccessViolationException)
                        {
                            // Retry Nb. 2:
                            PythonEngine.Initialize();
                        }
                    }

                    PythonEngine.BeginAllowThreads();

                    string path = PythonEngine.PythonPath;
                    string home = PythonEngine.PythonHome;

                    using (Py.GIL())
                    {

                        dynamic pymod = Py.Import("mixemode");
                        logger.Debug("Imported testing module");
                        if ((int)pymod.DEBUG == 1)
                        {
                            AttachDebugger(); // enable this for mixed-mode debugging
                        }

                        logger.Debug(nameof(InitPython), "Python engine initialized successfully.");
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Error(nameof(InitPython), "Unable to initialize python engine", ex);
            }
        }


        public static Pythonet_2prod GetInstanceForUser(int userId, int nbDiners)
        {
            try
            {
                RemoveOldUserWrapperInstances();
                if (instancePerUser.ContainsKey(userId))
                {
                    return instancePerUser[userId].Item2;
                }

                if (nbDiners <= 0)
                {
                    // Search for NbDiners...
                    //
                    using (ShopDBModel context = new ShopDBModel())
                    {
                        TableUser tu = context.TableUsers
                            .OrderByDescending(t => t.AssociationTimestamp)
                            .FirstOrDefault(t => t.PersonID == userId);
                        if (tu != null)
                        {
                            int deviceId = tu.DeviceID;

                            // Firstly on the OrderHeader...
                            OrderHeader orderHeader = context.OrderHeaders.Where(oh => oh.DeviceID == deviceId).OrderByDescending(oh => oh.ID).FirstOrDefault();
                            if (orderHeader != null)
                            {
                                nbDiners = orderHeader.NbDiners;
                            }
                            else
                            {
                                // And secondly, if not found, on the Device itself.
                                int? nullableNbDiners = context.SmartTables.Where(d => d.ID == deviceId).Select(d => d.NbDiners).FirstOrDefault();
                                if (nullableNbDiners.HasValue)
                                {
                                    nbDiners = nullableNbDiners.Value;
                                }
                            }
                        }

                        if (nbDiners == 0)
                        {
                            logger.Error(nameof(GetInstanceForUser), $"Unable to get instance for user with id {userId} because nbDiners <= 0");
                            return null;
                        }
                    }
                }

                // Create an instance for this unknown user:
                Pythonet_2prod pythonWrapper = new Pythonet_2prod();
                pythonWrapper.InitUser(userId, DateTime.Now, nbDiners);
                instancePerUser.Add(userId, new Tuple<DateTime, Pythonet_2prod>(DateTime.Now, pythonWrapper));
                return pythonWrapper;
            }
            catch (Exception ex)
            {
                logger.Error(nameof(GetInstanceForUser), "Unable to get instance for user with id " + userId, ex);
                return null;
            }
        }

        private static void RemoveOldUserWrapperInstances()
        {
            try
            {
                DateTime timeThreshold = DateTime.Now.Subtract(new TimeSpan(2, 0, 0));

                List<int> userIdsToDelete = new List<int>();
                foreach(var kvp in instancePerUser)
                {
                    if (kvp.Value.Item1 < timeThreshold)
                    {
                        // Mark this item as to be deleted:
                        userIdsToDelete.Add(kvp.Key);
                    }
                }

                foreach(var key in userIdsToDelete)
                {
                    instancePerUser.Remove(key);
                }
            }
            catch (Exception ex)
            {
                logger.Error(nameof(RemoveOldUserWrapperInstances), "Unable to remove old python wrapper instances", ex);
            }
        }

        /// <summary>
        /// One instance per user
        /// </summary>
        /// <param name="userId"></param>
        /// <param name="timestamp"></param>
        /// <param name="something"></param>
        public void InitUser(int userId, DateTime timestamp, int nbDiners)
        {
            try
            {
                InitPython();

                this.userId = userId;
                this.timestamp = timestamp;
                this.nbDiners = nbDiners;

                // Log features:
                logger.Debug($"NbDiners is: {nbDiners}, UserID is: {userId}, CreationDatetime is: {timestamp}");

                using (Py.GIL())
                {

                    // Launching the script called "pred2prod.py" which contains the algorithm written under python
                    this.pythonModule = Py.Import("pred2prod");  // <---- @TODO: keep one instance per user!!!

                    DateTime CreationDatetime = this.timestamp;
                    int epoch = (int)(CreationDatetime - new DateTime(1970, 1, 1)).TotalSeconds-300;// time now - 1200 secondes to simulate customer's  visiting time

                    logger.Debug("epoch = " + epoch.ToString());

                    // calling the function 'testtime' in python and converting the variable 'epoch' to datetime type under python 
                    dynamic tm = pythonModule.testtime(epoch);
                    logger.Debug("CreationDateTime: " + tm);

                    getFeatures = pythonModule.Feature(this.nbDiners, this.userId, tm);// replace CreationDatetime by 'tm'

                    // Getting  user profile vector
                    userProfile = getFeatures.get_user_profile();
                    logger.Debug("User profile:");
                    Console.Write(userProfile);
                }
            }
            catch (Exception ex)
            {
                logger.Error(nameof(InitUser), "Unable to initialize python wrapper instance", ex);
            }
        }

        public void AddEvent(Event evt)
        {
            try
            {

                if (evt == null)
                {
                    return;
                }

                using (Py.GIL())
                {
                    string newParameter = evt.Parameter;

                    // simulating & assigning values to features
                    logger.Debug($"Received event: parameter = {newParameter}");

                    // Getting event vector
                    dynamic ae = getFeatures.add_Event(newParameter);
                    logger.Debug("new event:" + ae);
                    Console.WriteLine("EventList:" + ae);
                }
            }
            catch (Exception ex)
            {
                logger.Error(nameof(AddEvent), "Unable to consume user event", ex);
            }
        }

        public Dictionary<int, double> AddEventsAndGetPredictions(Event evt)
        {
            try
            {
                AddEvent(evt);
                return GetPredictions();
            }
            catch (Exception ex)
            {
                logger.Error(nameof(AddEventsAndGetPredictions), "Unable to add user event and compute user predictions", ex);
                return null;
            }
        }

        static void AttachDebugger()
        {
            logger.Debug("waiting for .NET debugger to attach");
            while (!Debugger.IsAttached)
            {
                Thread.Sleep(100);
            }
            logger.Debug(".NET debugger is attached");

        }

        /// <summary>
        /// Key-Value : ProductId - Prediction Value (rate)
        /// </summary>
        /// <returns></returns>
        public Dictionary<int, double> GetPredictions()
        {
            try
            {
                using (Py.GIL())
                {
                    dynamic ev = getFeatures.Event2vec();
                    logger.Debug("Event vector:" + ev?.ToString());
                    Console.WriteLine("Event vector:" + ev);

                    //simulate a time now
                    int epoch1 = (int)(DateTime.Now - new DateTime(1970, 1, 1)).TotalSeconds + 1200;
                    Console.WriteLine("epoch:" + epoch1);
                    dynamic timenow1 = pythonModule.testtime(epoch1);
                    logger.Debug("Timenow1:" + timenow1?.ToString());
                    Console.WriteLine("Timenow1:" + timenow1);

                    dynamic td = getFeatures.get_timedelta();
                    logger.Debug(" Time delta" + td?.ToString());
                    Console.WriteLine("Time delta" + td);
                    // calling and showing fit parameters of the algorithm
                    dynamic W1 = pythonModule.W1;
                    logger.Debug("W1: " + W1?.ToString());
                    dynamic W2 = pythonModule.W2;
                    logger.Debug("W2: " + W2?.ToString());
                    dynamic W3 = pythonModule.W3;
                    logger.Debug("W3: " + W3?.ToString());

                    // importing the fit matrix ' item_context_pred" founded under python code
                    dynamic ict = pythonModule.item_context_pred;
                    logger.Debug("predicted context: " + ict?.ToString());

                    // importing the fit matrix ' Products" founded under python code
                    dynamic products = pythonModule.Products;
                    logger.Debug("Products: " + ict?.ToString());

                    // checking python feature engineering functions before fitting inputs to the model

                    // input derived from contexts
                    dynamic ctoi = pythonModule.Context2Item(ict);
                    logger.Debug("Sale probability for each item  according to the actual context: " + ctoi?.ToString());

                    // input derived from user profile
                    dynamic uti = pythonModule.User2Item(this.userProfile);
                    logger.Debug("Sale probability for each item  according to the user_profile: " + uti?.ToString());

                    // input derived from happening events
                    dynamic eti = pythonModule.Event2Item(ev);
                    logger.Debug("Sale probability for each item  according to happening event: " + eti?.ToString());
                    Console.WriteLine("Sale probability for each item  according to happening event: " + eti?.ToString());

                    // input derived from time delta
                    dynamic mah = pythonModule.Mah(td);
                    logger.Debug("Sale probability for each item  according to the visit duration of the client: " + mah?.ToString());
                    Console.WriteLine("Sale probability for each item  according to the visit duration of the client: " + mah?.ToString());


                    // putting all together and using the python class "Recommendation" to make predictions
                    dynamic rec = pythonModule.Recommendation(ict, products, userProfile, ev, td, W1, W2, W3);

                    dynamic pred;
                    if (UsedPredictionMode == PredictionMode.PredictionWithProductScoreOptimization)
                    {
                        logger.Debug("SCORE-BASED PREDICTION:");
                        pred = rec.get_prediction(out_format:"dict");// out_format: " dict" gives the output under a dictionary
                        Console.WriteLine("SCORE-BASED PREDICTION:");
                        Console.WriteLine(pred);
                        //dynamic predtodict = pred.to_dict();
                        
                    }
                    else
                    {
                        logger.Debug("PRICE-BASED PREDICTION:");
                        pred = rec.get_optprediction();// out_format: " dict" gives the uoutput under a dictionary
                        Console.WriteLine("PRICE-BASED PREDICTION:");
                        Console.WriteLine(pred);
                        //dynamic predtodict = pred.to_dict();
                    }

                    if (pred != null)
                    {
                        logger.Debug(pred?.ToString());
                    }

                    var converter = PyConverter.NewConverter();  //create an instance of PyConverter, take from ConverteurPy.cs script
                    Dictionary<string, object> productNameAndScores = converter.Convert(pred); //b is a List of CLR objects

                    //
                    // Transform product Name into equivalent product Id:
                    //

                    Dictionary<int, double> productIdAndScores = new Dictionary<int, double>();
                    if (productNameAndScores != null)
                    {
                        using (ShopDBModel context = new ShopDBModel())
                        {
                            //List<ProductScore> productScores = productNameAndScores.Select(pns => new ProductScore { Name = pns.Key, Score = pns.Value, Id = 0 }).ToList();
                            //List<string> productNames = productScores.Select(ps => ps.Name).ToList();

                            //// Retrieve corresponding products from database:
                            //List<ProductScore> productsFromDB = context.Products.Where(p => productNames.Contains(p.Name)).Select(p2 => new ProductScore { Id = p2.ID, Name = p2.Name }).ToList();
                            //ProductScore currentProductScore = null;

                            //// Change product Name with product ID:
                            //foreach (KeyValuePair<string, double> kvp in productNameAndScores)
                            //{
                            //    currentProductScore = productsFromDB.FirstOrDefault(pfd => pfd.Name == kvp.Key);
                            //    if (currentProductScore == null)
                            //    {
                            //        continue;
                            //    }
                            //    productIdAndScores.Add(currentProductScore.Id, kvp.Value);
                            //}
                        }
                    }
                    return productIdAndScores;
                }
            }
            catch (Exception ex)
            {
                logger.Error(nameof(GetPredictions), "Unable to compute user predictions", ex);
                return null;
            }

        }
    }

    internal class ProductScore
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public double Score { get; set; }
    }
}

