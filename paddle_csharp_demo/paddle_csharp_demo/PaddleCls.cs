using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace paddle_cshap_demo
{
    /// <summary>
    /// PaddleCls
    /// 二次封装paddle deploy,提取出clas模型相关流程
    /// 针对当前项目进行了修改部分实现
    /// </summary>
    class PaddleCls
    {
        public List<string> imagePaths = null;
        public List<string> modelFiles = null;
        public MetaInformation metaInfos = new MetaInformation();
        public IntPtr model;
        public List<Task> tasks = new List<Task>();

        //label分类描述的最大长度
        public static int MaxStrLen = 10;

        public bool use_gpu = false;
        public byte[] paddlex_model_type = new byte[10];

        //从目录加载模型
        public async Task LoadModel(string model_path, string model_type = "paddlex") 
        {
            LoadInfo("./model/model_dy");

            metaInfos.ModelType = model_type;
            metaInfos.NumModels = 1;
            metaInfos.GpuId = 0;



            if (metaInfos.ModelType == null) metaInfos.Log += $"[Error] model type is Null.\r\n";
            else if (metaInfos.NumModels <= 0) metaInfos.Log += $"[Error] model threads is <=0.\r\n";
            else if (metaInfos.ModelDirs.Count == 0) metaInfos.Log += $"[Error] model dir is Null.\r\n";
            else if (metaInfos.GpuId < 0 || metaInfos.GpuId > 7) metaInfos.Log += $"[Error] GPU id is not valid.\r\n";
            else
            {
                await InitModelAsync(metaInfos.ModelDirs[0]);

            }
        }

        //从目录加载模型图片
        public void LoadImg(string path)
        {
            try
            {

                metaInfos.ImgDir = path;
                metaInfos.Log = metaInfos.Log + "[Info] Test Image Dir: " + metaInfos.ImgDir + "\r\n";

                if (Directory.Exists(metaInfos.ImgDir))
                {
                    imagePaths = Directory.GetFiles(metaInfos.ImgDir).ToList<string>();
                    metaInfos.Log = metaInfos.Log + $"[Info] Found {imagePaths.Count()} image files.\r\n";
                }
                else
                {
                    System.Windows.Forms.MessageBox.Show("Error: Can not find any image files！");
                }

            }
            catch (Exception ex)
            {
                throw new Exception($"Choose Image Dir Error: {ex.Message}");
            }
        }

        
        public void LoadInfo(string inputDir)
        {
            if (Directory.GetFiles(inputDir).Length > 0)
            {
                metaInfos.Log = metaInfos.Log + "[Info] Chosen 1 Model Dir\r\n";
                modelFiles = Directory.GetFiles(inputDir).ToList<string>();

                metaInfos.ModelDirs.Add(inputDir);
                metaInfos.ModelFiles.Add(modelFiles);

                metaInfos.Log = metaInfos.Log + "    Dir: " + inputDir + "\r\n";
                for (int i = 0; i < modelFiles.Count(); i++)
                {
                    metaInfos.Log = metaInfos.Log + $"        {modelFiles[i]}\r\n";
                }
            }
        }

        //初始化模型
        public async Task InitModelAsync(string model_dir)
        {

            await Task.Run(() =>
            {

                metaInfos.IsReady = false;
                metaInfos.NeedUpdate = false;

                // 创建新模型
                for (int i = 0; i < metaInfos.NumModels; i++)
                {
                    // 模型初始化
                    IntPtr minInputSize = IntPtr.Zero;
                    IntPtr maxInputSize = IntPtr.Zero;
                    IntPtr optInputSize = IntPtr.Zero;
                    IntPtr paddlex_model_type_ = Marshal.AllocHGlobal(10);
                    try
                    {
                        if (!metaInfos.UseTrt)
                        {
                            IntPtr model_ = InferModel.ModelObjInit(metaInfos.ModelType, model_dir, use_gpu, metaInfos.GpuId, metaInfos.UseTrt, paddlex_model_type_);

                            model = model_;
                            metaInfos.Log = metaInfos.Log + $"[Info] Finished create [No.{i + 1}] {metaInfos.ModelType} model from {model_dir}\r\n";
                        }
                        else
                        {
                            metaInfos.Log = metaInfos.Log + $"[Info] Creating models with TensorRT acceleration will take a long long time, please wait with patient...\r\n";
                            //Thread.Sleep(500);
                            int[] minInputSize_ = new int[4] { 1, 3, metaInfos.MinH, metaInfos.MinW };
                            int[] maxInputSize_ = new int[4] { 1, 3, metaInfos.MaxH, metaInfos.MaxW };
                            int[] optInputSize_ = new int[4] { 1, 3, metaInfos.OptH, metaInfos.OptW };
                            minInputSize = Marshal.AllocHGlobal(4 * 4);
                            maxInputSize = Marshal.AllocHGlobal(4 * 4);
                            optInputSize = Marshal.AllocHGlobal(4 * 4);
                            Marshal.Copy(minInputSize_, 0, minInputSize, 4);
                            Marshal.Copy(maxInputSize_, 0, maxInputSize, 4);
                            Marshal.Copy(optInputSize_, 0, optInputSize, 4);

                            //IntPtr model_ = InferModel.ModelObjInit(metaInfos.ModelType, model_dir, metaInfos.GpuId, metaInfos.UseTrt, paddlex_model_type_,
                            //    minInputSize, maxInputSize, optInputSize, metaInfos.Precision, metaInfos.MinSubgraphSize);
                            metaInfos.Log = metaInfos.Log + $"[Info] Collecting trt model shape range info and save to {metaInfos.ShapeRangeInfoPath}...\r\n";
                            IntPtr model_ = InferModel.ModelObjInit(metaInfos.ModelType, model_dir, use_gpu, metaInfos.GpuId, metaInfos.UseTrt, paddlex_model_type_,
                                minInputSize, maxInputSize, optInputSize, metaInfos.Precision, metaInfos.MinSubgraphSize,
                                metaInfos.TargetWidth, metaInfos.TargetHeight, metaInfos.ShapeRangeInfoPath);

                            model = model_;
                            metaInfos.Log = metaInfos.Log + $"[Info] Finished create [No.{i + 1}] {metaInfos.ModelType} with TensorRT acceleration model!\r\n";
                        }
                        // paddlex获取模型实际类型并更新ModelType
                        if (metaInfos.ModelType == "paddlex")
                        {
                            Marshal.Copy(paddlex_model_type_, paddlex_model_type, 0, 10);
                            string tmp = System.Text.Encoding.ASCII.GetString(paddlex_model_type);
                            metaInfos.ModelType = tmp.Split(new char[] { '\0' })[0];
                        }
                    }
                    catch (Exception ex)
                    {
                        throw ex;
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(paddlex_model_type_);
                        if (metaInfos.UseTrt)
                        {
                            Marshal.FreeHGlobal(minInputSize);
                            Marshal.FreeHGlobal(maxInputSize);
                            Marshal.FreeHGlobal(optInputSize);
                        }
                    }
                }
                metaInfos.IsReady = true;
            });
        }




        //异步推理指定目录中的图片
        public async Task PredictDirAsync(string imageDir)
        {
            if (!metaInfos.IsReady)
                return;

            LoadImg(imageDir);

            await Task.Run(() =>
            {

                //添加任务列表
                for (int i = 0; i < imagePaths.Count(); i++)
                {
                    Mat src = Cv2.ImRead(imagePaths[i], ImreadModes.Unchanged);
                    // 调整图片尺寸(实际部署可省略)
                    int srcW = src.Cols;
                    int srcH = src.Rows;
                    double scale_factor = 0.0;
                    ResumeParam param = new ResumeParam();
                    if (metaInfos.TargetHeight != 0 && metaInfos.TargetWidth != 0)
                    {
                        ImageHelper.rescale(src, ref src, metaInfos.TargetWidth, metaInfos.TargetHeight, ref scale_factor);
                        param.scale_factor = scale_factor;
                        param.src_height = srcH;
                        param.src_width = srcW;
                    }


                    int h = src.Rows;
                    int w = src.Cols;
                    int c = src.Channels();

                    tasks.Add(CreateTask(metaInfos.ModelType, model, src.Data, i, w, h, c, param));

                }
                //开始任务
                for (int j = 0; j < tasks.Count; j++)
                {
                    tasks[j].Start();
                }
                try
                {
                    Task.WaitAll(tasks.ToArray());
                    metaInfos.Log = metaInfos.Log + $"[Info] predicted {tasks.Count} images.\r\n";
                }
                catch (Exception ex)
                {
                    throw ex;
                }
                tasks.Clear();

            });
        }

        //进行一次推理
        public async Task<ClsResult> PredictOneAsync(Bitmap bmp)
        {
            Mat src = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);
            // 调整图片尺寸(实际部署可省略)
            int srcW = src.Cols;
            int srcH = src.Rows;
            double scale_factor = 0.0;
            ResumeParam param = new ResumeParam();
            if (metaInfos.TargetHeight != 0 && metaInfos.TargetWidth != 0)
            {
                ImageHelper.rescale(src, ref src, metaInfos.TargetWidth, metaInfos.TargetHeight, ref scale_factor);
                param.scale_factor = scale_factor;
                param.src_height = srcH;
                param.src_width = srcW;
            }


            int h = src.Rows;
            int w = src.Cols;
            int c = src.Channels();


            ClsResult result = await PredictClsAsync(model, src.Data, 0, w, h, c);

            return result;
        }

        public async Task<ClsResult> PredictOneAsync(string img_path)
        {

            Mat src = Cv2.ImRead(img_path);
            // 调整图片尺寸(实际部署可省略)
            int srcW = src.Cols;
            int srcH = src.Rows;
            double scale_factor = 0.0;
            ResumeParam param = new ResumeParam();
            if (metaInfos.TargetHeight != 0 && metaInfos.TargetWidth != 0)
            {
                ImageHelper.rescale(src, ref src, metaInfos.TargetWidth, metaInfos.TargetHeight, ref scale_factor);
                param.scale_factor = scale_factor;
                param.src_height = srcH;
                param.src_width = srcW;
            }


            int h = src.Rows;
            int w = src.Cols;
            int c = src.Channels();


            ClsResult result = await PredictClsAsync(model, src.Data, 0, w, h, c);

            return result;
        }




        //执行推理
        public async Task<ClsResult> PredictClsAsync(IntPtr modelObj, IntPtr imageData, int id, int width, int height, int channels)
        {

            IntPtr score = IntPtr.Zero;
            IntPtr category = IntPtr.Zero;
            IntPtr category_id = IntPtr.Zero;
            try
            {
                score = Marshal.AllocHGlobal(4);  // float
                category = Marshal.AllocHGlobal(MaxStrLen); // 10个char
                category_id = Marshal.AllocHGlobal(4); // int

                //cls推理：score表示该预测类别得分(0-1)，category_id表示类别id，category表示类别的字符串描述
                await Task.Run(() => InferModel.ModelObjPredict_Cls(modelObj, imageData, width, height, channels, score, category, category_id));


                float[] _score = new float[1];
                byte[] _category = new byte[MaxStrLen];
                int[] _category_id = new int[1];
                Marshal.Copy(score, _score, 0, 1);
                Marshal.Copy(category, _category, 0, MaxStrLen);
                Marshal.Copy(category_id, _category_id, 0, 1);

                //所有label都需要有终止符"."，用来分割结果
                string labelDesc = System.Text.Encoding.ASCII.GetString(_category).Split('.')[0];

                metaInfos.Log = metaInfos.Log + $"[Info] image id: {id} , category_id: {_category_id[0]}, score: {_score[0]}," +
                    $" category: {labelDesc}\r\n";

                return new ClsResult(_category_id[0], labelDesc);

            }
            catch (Exception ex)
            {
                throw ex;
            }
            finally
            {
                Marshal.FreeHGlobal(score);
                Marshal.FreeHGlobal(category);
                Marshal.FreeHGlobal(category_id);
            }

        }

        // 分类推理 IntPtr score, IntPtr category, IntPtr category_id
        public ClsResult Callback_Predict_Cls(IntPtr modelObj, IntPtr imageData, int id, int width, int height, int channels)
        {
            IntPtr score = IntPtr.Zero;
            IntPtr category = IntPtr.Zero;
            IntPtr category_id = IntPtr.Zero;
            try
            {
                score = Marshal.AllocHGlobal(4);  // float
                category = Marshal.AllocHGlobal(MaxStrLen); // 10个char
                category_id = Marshal.AllocHGlobal(4); // int


                //cls推理：score表示该预测类别得分(0-1)，category_id表示类别id，category表示类别的字符串描述
                InferModel.ModelObjPredict_Cls(modelObj, imageData, width, height, channels, score, category, category_id);


                float[] _score = new float[1];
                byte[] _category = new byte[MaxStrLen];
                int[] _category_id = new int[1];
                Marshal.Copy(score, _score, 0, 1);
                Marshal.Copy(category, _category, 0, MaxStrLen);
                Marshal.Copy(category_id, _category_id, 0, 1);

                //所有label都需要有终止符"."，用来分割结果
                string labelDesc = System.Text.Encoding.ASCII.GetString(_category).Split('.')[0];

                metaInfos.Log = metaInfos.Log + $"[Info] image id: {id} , category_id: {_category_id[0]}, score: {_score[0]}," +
                    $" category: {labelDesc}\r\n";

                return new ClsResult(_category_id[0], labelDesc);

            }
            catch (Exception ex)
            {
                throw ex;
            }
            finally
            {
                Marshal.FreeHGlobal(score);
                Marshal.FreeHGlobal(category);
                Marshal.FreeHGlobal(category_id);
            }
        }

        public Task<ClsResult> CreateTask(string taskType, IntPtr modelObj, IntPtr imageData, int id, int width, int height, int channels, ResumeParam param)
        {
            if (taskType == "clas")
                return new Task<ClsResult>(() => Callback_Predict_Cls(modelObj, imageData, id, width, height, channels));

            else
                throw new System.Exception($"[Error] taskType can only be: clas/seg/det/mask, but got {taskType}");
        }

        public void Destruct()
        {
            metaInfos.IsReady = false; 
            InferModel.ModelObjDestruct(model);
        }
    }

    class ClsResult 
    { 
        public int category_id; //类别id
        public string category; //类别描述
        public ClsResult(int _category_id, string _category) 
        {
            category_id = _category_id;
            category = _category;
        }

    }
}
