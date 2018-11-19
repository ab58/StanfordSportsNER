import java.util.*;
import java.io.*;
import java.nio.file.Files;

public class TSVConvert {

    public static void main(String[] args) throws Exception {

        File[] files = new File(args[0]).listFiles();
        PrintWriter tsv = new PrintWriter(args[1]);

        for (File file : files) {
            Scanner fileIn = new Scanner(file);
            while (fileIn.hasNextLine()) {
                StringTokenizer st = new StringTokenizer(fileIn.nextLine(), " ");
                while (st.hasMoreTokens()) {
                    tsv.println(st.nextToken()+"\tO");
                }
                tsv.println();
            }
            fileIn.close();
        }
        tsv.close();
    }
}
