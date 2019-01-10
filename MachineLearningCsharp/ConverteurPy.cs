using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Python.Runtime;


namespace MachineLearningCsharp
{
    public class PyConverter
    {
        public PyConverter()
        {
            this.Converters = new Dictionary<IntPtr, Func<PyObject, object>>();
        }

        private Dictionary<IntPtr, Func<PyObject, object>> Converters;

        public void Add(IntPtr type, Func<PyObject, object> func)
        {
            this.Converters.Add(type, func);
        }

        public object Convert(PyObject obj)
        {
            if (obj == null)
            {
                return null;
            }
            PyObject type = obj.GetPythonType();
            Func<PyObject, object> func;
            var state = Converters.TryGetValue(type.Handle, out func);
            if (!state)
            {
                if ((type + "").Contains("float64"))
                {
                    state = Converters.TryGetValue(PythonEngine.Eval("float").Handle, out func);
                    if (!state)
                    {
                        throw new Exception($"Type {type.ToString()} not recognized (1)");
                    }
                }
                else
                {
                    throw new Exception($"Type {type.ToString()} not recognized (2)");
                }
            }
            return func(obj);
        }


        public  static dynamic NewConverter()
        {
            var converter = new PyConverter();
            using (Py.GIL())
            {
                //XIncref is needed, or keep the PyObject
                converter.Converters.Add(PythonEngine.Eval("int").Handle, (args) => { return args.AsManagedObject(typeof(int)); });
                converter.Converters.Add(PythonEngine.Eval("str").Handle, (args) => { return args.AsManagedObject(typeof(string)); });
                converter.Converters.Add(PythonEngine.Eval("float").Handle, (args) => { return args.AsManagedObject(typeof(double)); });
                converter.Converters.Add(PythonEngine.Eval("bool").Handle, (args) => { return args.AsManagedObject(typeof(bool)); });
                converter.Converters.Add(PythonEngine.Eval("list").Handle, (args) => 
                {

                    List<object> list = new List<object>();
                    for (int i = 0; i < args.Length(); i++)
                    {
                        dynamic newVal=converter.Convert(args[i]);
                        list.Add(newVal);
                    }
                    return list;
                });
                converter.Converters.Add(PythonEngine.Eval("dict").Handle, (args) =>
                {
                    Dictionary<string, object> dict = new Dictionary<string, object>();
                    var dictionaryItems = args.InvokeMethod("items");
                    Console.WriteLine("dictionaryItems");
                    foreach (dynamic keyValue in dictionaryItems)
                    {
                        dynamic newKey = converter.Convert(keyValue[0]);
                        dynamic newValue = converter.Convert(keyValue[1]);
                        dict.Add(newKey.ToString(),newValue);
                    }
                    return dict;
                });
            }
            return converter;
        }
         
        
    }

}
