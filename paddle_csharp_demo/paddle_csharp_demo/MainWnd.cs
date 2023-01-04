using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace paddle_cshap_demo
{
    public partial class MainWnd : Form
    {
        PaddleCls paddleCls = new PaddleCls();
        public MainWnd()
        {
            InitializeComponent();
        }

        private async void button1_Click(object sender, EventArgs e)
        {
            await paddleCls.LoadModel("./model/model_dy");
        }

        private async void button2_Click(object sender, EventArgs e)
        {
            await paddleCls.PredictDirAsync("./model/images");
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            textBox1.SelectionStart = textBox1.Text.Length;
            textBox1.SelectionLength = 0;
            textBox1.ScrollToCaret();
        }

        private void MainWnd_Load(object sender, EventArgs e)
        {
            textBox1.DataBindings.Add("Text", paddleCls.metaInfos, "Log", false, DataSourceUpdateMode.Never);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            paddleCls.Destruct();
        }


        private async void button4_Click(object sender, EventArgs e)
        {
            ClsResult result = await paddleCls.PredictOneAsync("./model/images/DefectImage10.bmp");
            Console.WriteLine(result.category);
        }
    }
}
