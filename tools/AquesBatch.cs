using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;

// Batch AquesTalk (bouyomichan "female1" = Reimu) synthesizer.
// usage: AquesBatch.exe <list.tsv> <outDir> [errLog]
// list.tsv (UTF-8), one item per line:  id<TAB>speed<TAB>koe-text
// Writes <outDir>/<id>.wav (8kHz mono 16bit) per item; failures go to errLog as: id<TAB>errcode
class AquesBatch
{
    [DllImport("AquesTalk.dll", CallingConvention = CallingConvention.StdCall)]
    static extern IntPtr AquesTalk_Synthe(byte[] koe, int iSpeed, out int size);

    [DllImport("AquesTalk.dll", CallingConvention = CallingConvention.StdCall)]
    static extern void AquesTalk_FreeWave(IntPtr wav);

    static int Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("usage: AquesBatch.exe <list.tsv> <outDir> [errLog]");
            return 1;
        }
        string listPath = args[0];
        string outDir = args[1];
        string errPath = args.Length > 2 ? args[2] : Path.Combine(outDir, "_errors.tsv");
        Directory.CreateDirectory(outDir);

        Encoding sjis = Encoding.GetEncoding(932);
        int ok = 0, fail = 0;
        using (StreamWriter err = new StreamWriter(errPath, false, new UTF8Encoding(false)))
        {
            foreach (string line in File.ReadLines(listPath, new UTF8Encoding(false)))
            {
                if (line.Length == 0) continue;
                string[] parts = line.Split(new char[] { '\t' }, 3);
                if (parts.Length < 3) continue;
                string id = parts[0];
                int speed = int.Parse(parts[1]);
                byte[] body = sjis.GetBytes(parts[2]);
                byte[] koeBuf = new byte[body.Length + 1];
                Array.Copy(body, koeBuf, body.Length);

                int size;
                IntPtr wav = AquesTalk_Synthe(koeBuf, speed, out size);
                if (wav == IntPtr.Zero)
                {
                    err.WriteLine(id + "\t" + size);
                    fail++;
                    continue;
                }
                byte[] buf = new byte[size];
                Marshal.Copy(wav, buf, 0, size);
                AquesTalk_FreeWave(wav);
                File.WriteAllBytes(Path.Combine(outDir, id + ".wav"), buf);
                ok++;
            }
        }
        Console.WriteLine("done ok=" + ok + " fail=" + fail);
        return 0;
    }
}
