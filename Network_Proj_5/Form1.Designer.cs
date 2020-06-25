namespace Network_Proj_5
{
    partial class WeatherAppForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.ZC_Lbl = new System.Windows.Forms.Label();
            this.Name_Lbl = new System.Windows.Forms.Label();
            this.GWBZ_Button = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // ZC_Lbl
            // 
            this.ZC_Lbl.AutoSize = true;
            this.ZC_Lbl.Location = new System.Drawing.Point(36, 53);
            this.ZC_Lbl.Name = "ZC_Lbl";
            this.ZC_Lbl.Size = new System.Drawing.Size(50, 13);
            this.ZC_Lbl.TabIndex = 0;
            this.ZC_Lbl.Text = "Zip Code";
            // 
            // Name_Lbl
            // 
            this.Name_Lbl.AutoSize = true;
            this.Name_Lbl.Location = new System.Drawing.Point(36, 22);
            this.Name_Lbl.Name = "Name_Lbl";
            this.Name_Lbl.Size = new System.Drawing.Size(35, 13);
            this.Name_Lbl.TabIndex = 1;
            this.Name_Lbl.Text = "Name";
            // 
            // GWBZ_Button
            // 
            this.GWBZ_Button.Location = new System.Drawing.Point(67, 307);
            this.GWBZ_Button.Name = "GWBZ_Button";
            this.GWBZ_Button.Size = new System.Drawing.Size(248, 21);
            this.GWBZ_Button.TabIndex = 2;
            this.GWBZ_Button.Text = "button1";
            this.GWBZ_Button.UseVisualStyleBackColor = true;
            // 
            // WeatherAppForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(388, 384);
            this.Controls.Add(this.GWBZ_Button);
            this.Controls.Add(this.Name_Lbl);
            this.Controls.Add(this.ZC_Lbl);
            this.Name = "WeatherAppForm";
            this.Text = "Weather Application (Project 5)";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label ZC_Lbl;
        private System.Windows.Forms.Label Name_Lbl;
        private System.Windows.Forms.Button GWBZ_Button;
    }
}

