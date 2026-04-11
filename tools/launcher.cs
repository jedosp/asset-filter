using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

class AssetFilterLauncher
{
    [STAThread]
    static int Main()
    {
        string exeDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');

        string pythonw = Path.Combine(exeDir, "python", "pythonw.exe");
        string python  = Path.Combine(exeDir, "python", "python.exe");
        string interp  = File.Exists(pythonw) ? pythonw : python;
        string script  = Path.Combine(exeDir, "src", "main.py");

        if (!File.Exists(interp))
        {
            MessageBox.Show(
                "python 폴더를 찾을 수 없습니다.\n압축 파일을 올바르게 해제했는지 확인하세요.",
                "Asset Filter",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 1;
        }

        var psi = new ProcessStartInfo
        {
            FileName = interp,
            Arguments = "\"" + script + "\"",
            WorkingDirectory = exeDir,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardError = true,
        };

        try
        {
            using (var proc = Process.Start(psi))
            {
                string stderr = proc.StandardError.ReadToEnd();
                proc.WaitForExit();

                if (proc.ExitCode != 0 && !string.IsNullOrWhiteSpace(stderr))
                {
                    if (stderr.Length > 2000)
                        stderr = stderr.Substring(stderr.Length - 2000);

                    MessageBox.Show(
                        "오류가 발생했습니다:\n\n" + stderr,
                        "Asset Filter",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error);
                }

                return proc.ExitCode;
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "실행 오류: " + ex.Message,
                "Asset Filter",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 1;
        }
    }
}
