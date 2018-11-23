import java.nio.file.Files;
import java.util.*;
import java.io.*;
import edu.stanford.nlp.ie.*;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.sequences.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.util.*;
import java.util.concurrent.atomic.*;
import java.util.stream.*;

public class StanfordNER {

    public static void deleteDirContents(String dirName) {
        File dir = new File(dirName);
        if (dir.isDirectory()) {
            for (File f : dir.listFiles()) {
                f.delete();
            }
        }
    }

    public static void trainAndWrite(String modelOutPath, String prop, String trainingFilepath) {
        Properties props = StringUtils.propFileToProperties(prop);
        props.setProperty("serializeTo", modelOutPath);

        //if input use that, else use from properties file.
        if (trainingFilepath != null) {
            props.setProperty("trainFile", trainingFilepath);
        }
        SeqClassifierFlags flags = new SeqClassifierFlags(props);
        CRFClassifier<CoreLabel> crf = new CRFClassifier<>(flags);
        crf.train();

        crf.serializeClassifier(modelOutPath);
    }

    public static CRFClassifier getModel(String modelPath) {
        return CRFClassifier.getClassifierNoExceptions(modelPath);
    }

    //Tagging method to call if you don't need it dumped to an output file
    public static String doTagging(CRFClassifier model, String input) {
        input = input.trim();
        String taggedString = model.classifyToString(input);
        return taggedString;
    }

    //Tagging method to call if an output file containing the tagged string is required
    public static String doTagging(CRFClassifier model, String input, PrintWriter out) {
        input = input.trim();
        String taggedString = model.classifyToString(input);
        out.println(taggedString);
        return taggedString;
    }

    public static String repeat(String s, int n) {
        String sRepeat = "";
        for (int i = 0; i < n; i++) {
            sRepeat = sRepeat+" "+s;
        }
        return sRepeat.trim();
    }

    public static void printSampleToResults(String sentence, String tags, PrintWriter resultsFile) {
        StringTokenizer sentTok = new StringTokenizer(sentence, " ");
        StringTokenizer tagTok = new StringTokenizer(tags, " ");
        while (sentTok.hasMoreTokens()) {
            resultsFile.print(sentTok.nextToken()+"/"+tagTok.nextToken());
            if (sentTok.hasMoreTokens()) {
                resultsFile.print(" ");
            }
        }
        resultsFile.println();
    }

    public static void printRankedMap(TreeMap<String,Integer> tm, PrintWriter resultsFile) {
        Stream<Map.Entry<String,Integer>> sorted = tm.entrySet().stream().sorted(Map.Entry.comparingByValue());
        Object[] sortedArr = sorted.toArray();
        for (int i = sortedArr.length-1; i >= 0; i--) {
            String res = sortedArr[i].toString().split("=")[0];
            String freq = sortedArr[i].toString().split("=")[1];
            resultsFile.println(res+"\t"+freq);
        }
    }

    public static double evaluateModel(String goldFilename, String outputDirname, CRFClassifier model, PrintWriter resultsFile) throws Exception {
        File goldFile = new File(goldFilename);
        File outputDir = new File(outputDirname);
        File[] files = outputDir.listFiles();
        Scanner goldFileIn = new Scanner(goldFile);
        int correct = 0;
        int incorrect = 0;
        int passes = 0;
        int fails = 0;
        int correctWithoutO = 0;
        int falsePosWithoutO = 0;
        int falseNegWithoutO = 0;
        //The following HashMap will track the true positives, false, positives,
        //and false negatives for every NER tags in the model. These numbers will
        //subsequently be used to calculate precision, recall, and f-scores.
        TreeMap<String,int[]> prfValues = new TreeMap<String,int[]>();
        TreeMap<String,Integer> emitters = new TreeMap<String,Integer>();
        TreeMap<String,Integer> attractors = new TreeMap<String,Integer>();
        TreeMap<String,Integer> emittersAttractors = new TreeMap<String,Integer>();
        for (File file:files) {
            //split each file into lines, and then each line into their
            //tagged tokens; then split each into token and tag
            Scanner testFileIn = new Scanner(file);
            String rawSentenceGold = "";
            String rawTagsGold = "";

            while (testFileIn.hasNextLine()) {
                String taggedLine = testFileIn.nextLine().replaceAll("\u00A0", " ");
                taggedLine = taggedLine.replaceAll("-LRB-", "(");
                taggedLine = taggedLine.replaceAll("-RRB-", ")");
                String[] taggedTokens = taggedLine.split("\\s+");
                //A line in the file will only show up in the gold file
                //if it has gold NER tags. If all words in a line are untagged,
                //this will not show up in the tsv; we have to account for this

                //we need to get the corresponding line in the gold file. To do this, we must
                //look at consecutive lines in the gold file until we hit a blank line.
                //The blank line will tell us that we've reached the end of the line.

                String rawSentence = "";
                String rawTags = "";
                for (String tt:taggedTokens) {
                    String token = tt;
                    String tag = "O";
                    if (tt.contains("/")) {
                        token = tt.substring(0, tt.lastIndexOf("/"));
                        tag = tt.substring(tt.lastIndexOf("/") + 1, tt.length());
                    }
                    rawSentence = rawSentence+" "+token;
                    rawTags = rawTags+" "+tag;
                }

                //System.out.println(model.classIndex.objectsList());
                for (Object ner_tag : model.classIndex.objectsList()) {
                    rawSentence = rawSentence.replaceAll("/"+ner_tag.toString(), "");
                }
                rawSentence = rawSentence.replaceAll("(\\w)[.][.]", "$1.");
                rawSentence = rawSentence.trim();
                rawTags = rawTags.trim();

                if (rawSentenceGold.equals("")) {
                    while (goldFileIn.hasNextLine()) {
                        String goldExample = goldFileIn.nextLine();
                        //System.out.println(goldExample);
                        //System.out.println(goldExample.length());
                        if (goldExample.equals("")) {
                            break;
                        }
                        //System.out.println("goldExample: "+goldExample);
                        String goldToken = goldExample.split("\t")[0];
                        String goldTag = goldExample.split("\t")[1];

                        rawSentenceGold = rawSentenceGold+" "+goldToken;
                        rawTagsGold = rawTagsGold+" "+goldTag;
                    }
                    rawSentenceGold = rawSentenceGold.trim();
                    rawTagsGold = rawTagsGold.trim();
                }

                //check if rawSentence is the same as rawSentenceGold; if it is
                //not, then we know this line is not covered in the gold file; in
                //this case, we must hold the goldFileIn Scanner at its current place;
                //this will keep rawSentenceGold and rawTagsGold the same until it sees
                //a rawSentence that is equal to rawSentenceGold. Also, we must change
                //rawSentenceGold to be rawSentence, and rawTagsGold to be "O" across
                //the board.

                System.out.println("\nrawSentence: "+rawSentence);
                System.out.println("rawTags: "+rawTags);
                System.out.println("rawSentenceGold: "+rawSentenceGold);
                System.out.println("rawTagsGold: "+rawTagsGold);
                String rsgEval = rawSentenceGold;
                String rtgEval = rawTagsGold;
                if (!rawSentence.equals(rawSentenceGold)) {
                    System.out.println("sentences are not the same");
                    rsgEval = rawSentence;
                    //System.out.println(rsgEval.split(" ").length);
                    rtgEval = repeat("O", rsgEval.split(" ").length);
                } else {
                    System.out.println("sentences are the same");
                    rawSentenceGold = "";
                    rawTagsGold = "";
                }
                System.out.println("rsgEval: "+rsgEval);
                System.out.println("rtgEval: "+rtgEval);
                resultsFile.print("\ngoldStandard: ");
                printSampleToResults(rsgEval, rtgEval, resultsFile);
                resultsFile.print("\nmodelOutput: ");
                printSampleToResults(rawSentence, rawTags, resultsFile);

                //to evaluate, we check on the TAGS. Evaluation is a simple
                //matter of comparing each tag in rawTags with its correspnding
                //tag in rtgEval. Matches score 1 for correct, mismatches score
                //1 for incorrect. Then divide by total comparisons.
                if (rawTags.equals(rtgEval)) {
                    resultsFile.println("\nPASS");
                    passes++;
                } else {
                    resultsFile.print("\nFAIL [");
                    fails++;
                }
                String[] rsgEvalArr = rsgEval.split(" ");
                String[] rawTagsArr = rawTags.split(" ");
                String[] rtgEvalArr = rtgEval.split(" ");
                int incorrectInLine = 0;
                for (int i = 0; i < rawTagsArr.length; i++) {
                    //System.out.println("raw tag: "+rawTagsArr[i]);
                    //System.out.println("gold tag: "+rtgEvalArr[i]);
                    //System.out.println(rawTagsArr[i].equals(rtgEvalArr[i]));

                    //check if each of these tags is contained in the map
                    //prfValues; if not, we insert with an empty Integer
                    //array of size 3; these will represent the number of
                    //true positives, false positives, and false negatives.
                    if (!prfValues.containsKey(rawTagsArr[i])) {
                        prfValues.put(rawTagsArr[i], new int[3]);
                    }
                    if (!prfValues.containsKey(rtgEvalArr[i])) {
                        prfValues.put(rtgEvalArr[i], new int[3]);
                    }

                    if (rawTagsArr[i].equals(rtgEvalArr[i])) {
                        correct++;
                        if (!rtgEvalArr[i].equals("O")) {
                            correctWithoutO++;
                        }
                        //increment true positive count for rawTagsArr[i] in prfValues
                        prfValues.get(rawTagsArr[i])[0]++;
                    } else {
                        incorrect++;
                        if (!rawTagsArr[i].equals("O")) {
                            falsePosWithoutO++;
                        }
                        if (!rtgEvalArr[i].equals("O")) {
                            falseNegWithoutO++;
                        }
                        incorrectInLine++;
                        String attractor = rawTagsArr[i];
                        String emitter = rtgEvalArr[i];
                        if (incorrectInLine > 1) {
                            resultsFile.print("; ");
                        }
                        resultsFile.print(rsgEvalArr[i]+"/"+emitter+"-->"+rsgEvalArr[i]+"/"+attractor);
                        //There are 2 types in "incorrect": false positives
                        //and false negatives. And incorrect prediction means
                        //it's a false positive for rawTagsArr[i] and a false
                        //negative for rtgEvalArr[i]. Update the false positive
                        //and false negative data accordingly
                        prfValues.get(attractor)[1]++;
                        prfValues.get(emitter)[2]++;
                        //Also in an incorrect case, track the specific failures,
                        //i.e. what the emittor is (class that yielded the false
                        //negative), what the attractor is (class that yielded
                        //the false positive), and the unique emittor-attractor
                        //pair. Increment these values in map to get counts.
                        if (!emitters.containsKey(emitter)) {
                            emitters.put(emitter, 0);
                        }
                        if (!attractors.containsKey(attractor)) {
                            attractors.put(attractor, 0);
                        }
                        if (!emittersAttractors.containsKey(emitter+"-->"+attractor)) {
                            emittersAttractors.put(emitter+"-->"+attractor, 0);
                        }
                        int incr = emitters.get(emitter).intValue()+1;
                        emitters.put(emitter, incr);
                        incr = attractors.get(attractor).intValue()+1;
                        attractors.put(attractor, incr);
                        incr = emittersAttractors.get(emitter+"-->"+attractor).intValue()+1;
                        emittersAttractors.put(emitter+"-->"+attractor, incr);
                    }
                }
                if (incorrectInLine > 0) {
                    resultsFile.println("]");
                }
                resultsFile.println("==================================================");
            }
        }

        //resultsFile.println("\n==================================================");
        resultsFile.println("\nLines with 100% correct tagging: "+passes);
        resultsFile.println("Lines with at least one failed tag: "+fails);

        resultsFile.println("\nMost Common Emitters");
        printRankedMap(emitters, resultsFile);

        resultsFile.println("\nMost Common Attractors");
        printRankedMap(attractors, resultsFile);


        resultsFile.println("\nMost Common Emitter-Attractor Pairs");
        printRankedMap(emittersAttractors, resultsFile);

        int sumTP = 0;
        int sumFP = 0;
        int sumFN = 0;
        double sumPrec = 0.0;
        double sumRec = 0.0;
        double sumF = 0.0;
        int numClasses = prfValues.keySet().size();

        System.out.println("\nEvaluation Metrics for Individual NER Tags:");
        System.out.println("Tag\tTruePos\tFalsePos\tFalseNeg\tPrecision\tRecall\tF-Score");
        resultsFile.println("\n\nEvaluation Metrics for Individual NER Tags:");
        resultsFile.println("Tag\tTruePos\tFalsePos\tFalseNeg\tPrecision\tRecall\tF-Score");

        for (String tag:prfValues.keySet()) {

            int[] prf = prfValues.get(tag);
            sumTP += prf[0];
            sumFP += prf[1];
            sumFN += prf[2];

            double precision = 0;
            double recall = 0;
            double fscore = 0;

            if (prf[0] != 0) {
                precision = (double)prf[0]/(prf[0]+prf[1]);
                recall = (double)prf[0]/(prf[0]+prf[2]);
                fscore = (precision*recall)/(precision+recall)*2;
            }

            sumPrec += precision;
            sumRec += recall;
            sumF += fscore;

            System.out.println(tag+"\t"+prf[0]+"\t"+prf[1]+"\t"+prf[2]+"\t"+precision+"\t"+recall+"\t"+fscore);
            resultsFile.println(tag+"\t"+prf[0]+"\t"+prf[1]+"\t"+prf[2]+"\t"+precision+"\t"+recall+"\t"+fscore);
        }
        System.out.println("ALL\t"+sumTP+"\t"+sumFP+"\t"+sumFN+"\t"+sumPrec/numClasses+"\t"+sumRec/numClasses+"\t"+sumF/numClasses);
        resultsFile.println("ALL\t"+sumTP+"\t"+sumFP+"\t"+sumFN+"\t"+sumPrec/numClasses+"\t"+sumRec/numClasses+"\t"+sumF/numClasses);

        System.out.println("\nTotal Correct Tags: "+correct);
        System.out.println("Total Incorrect Tags: "+incorrect);
        resultsFile.println("\nTotal Correct Tags: "+correct);
        resultsFile.println("Total Incorrect Tags: "+incorrect);
        resultsFile.println("Accuracy: "+(double)correct/(correct+incorrect));

        double precisionWithoutO = (double)correctWithoutO/(correctWithoutO+falsePosWithoutO);
        double recallWithoutO = (double)correctWithoutO/(correctWithoutO+falseNegWithoutO);
        double fScoreWithoutO = (precisionWithoutO*recallWithoutO)/(precisionWithoutO+recallWithoutO)*2;

        System.out.println("\nAccuracies of the \"true\" named entities (i.e. without the \"O\" tag)");
        System.out.println("Total Correct Tags: "+correctWithoutO);
        System.out.println("False Positives: "+falsePosWithoutO);
        System.out.println("False Negatives: "+falseNegWithoutO);
        System.out.println("Precision: "+precisionWithoutO);
        System.out.println("Recall: "+recallWithoutO);
        System.out.println("F-Score: "+fScoreWithoutO);

        resultsFile.println("\nAccuracies of the \"true\" named entities (i.e. without the \"O\" tag)");
        resultsFile.println("Total Correct Tags: "+correctWithoutO);
        resultsFile.println("False Positives: "+falsePosWithoutO);
        resultsFile.println("False Negatives: "+falseNegWithoutO);
        resultsFile.println("Precision: "+precisionWithoutO);
        resultsFile.println("Recall: "+recallWithoutO);
        resultsFile.println("F-Score: "+fScoreWithoutO);


        return (double)correct/(correct+incorrect);
    }


    public static void getTagResults(String dir, String trainOrTest, CRFClassifier trainedModel)  throws Exception {
        File folder = new File(dir);
        File[] files = folder.listFiles();
        for (File file:files) {
            File tagged_file = new File("tagged_files_"+trainOrTest+"/"+file.getName()+".tagged");
            tagged_file.getParentFile().mkdirs();
            PrintWriter out = new PrintWriter(tagged_file);
            Scanner in = new Scanner(file);
            while (in.hasNextLine()) {
                doTagging(trainedModel, in.nextLine(), out);
            }
            in.close();
            out.close();
        }
    }


    public static void main(String[] args) throws Exception {

        long startTime = System.currentTimeMillis();

        deleteDirContents("tagged_files_train");
        deleteDirContents("tagged_files_test");

        trainAndWrite(args[1]+".model", args[0], args[1]);
        CRFClassifier trainedModel = getModel(args[1]+".model");

        PrintWriter resultsTrain = new PrintWriter(args[5]);
        resultsTrain.println("Full Results Report: Training Data");
        resultsTrain.println("==================================================");
        PrintWriter resultsTest = new PrintWriter(args[6]);
        resultsTest.println("Full Results Report: Test Data");
        resultsTest.println("==================================================");

        //get fully tagged files for TRAINING SET
        getTagResults(args[2], "train", trainedModel);
        //get fully tagged files for TEST SET
        getTagResults(args[4], "test", trainedModel);

        System.out.println("\nTraining Accuracy: "+evaluateModel(args[1], "tagged_files_train", trainedModel, resultsTrain));
        System.out.println("\nTest Accuracy: "+evaluateModel(args[3], "tagged_files_test", trainedModel, resultsTest));
        resultsTrain.close();
        resultsTest.close();

        long endTime = System.currentTimeMillis();
        long seconds = (endTime - startTime) / 1000;
        long minutes = seconds / 60;
        seconds %= 60;
        System.out.println("\nRunning Time: "+minutes+" minutes "+seconds+" seconds");
    }
}
