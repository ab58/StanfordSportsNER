import java.util.*;
import java.io.*;
import java.util.concurrent.*;

import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.naturalli.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ie.util.*;

public class EventExtractor {

    public static ArrayList<String> getTags(String taggedString) {
        ArrayList<String> tagList = new ArrayList<String>();
        StringTokenizer st = new StringTokenizer(taggedString, " ");
        while(st.hasMoreTokens()) {
            String tag = st.nextToken().split("/")[1];
            tagList.add(tag);
        }
        return tagList;
    }

    public static boolean hasNamedEntity(ArrayList<String> tagList) {
        for (String tag : tagList) {
            if (!tag.equals("O")) {
                return true;
            }
        }
        return false;
    }

    public static List<RelationTriple> getShortAndLong(RelationTriple a, RelationTriple b) {
        List<RelationTriple> shortAndLong = new ArrayList<RelationTriple>();
        if (a.allTokens().size() <= b.allTokens().size()) {
            shortAndLong.add(a);
            shortAndLong.add(b);
        } else {
            shortAndLong.add(b);
            shortAndLong.add(a);
        }
        return shortAndLong;
    }

    public static class StringLengthSorter implements Comparator<RelationTriple> {

        @Override
        public int compare(RelationTriple a, RelationTriple b) {
            return a.allTokens().size() - b.allTokens().size();
        }
    }

    public static void eliminateShortenedEvents(Set<RelationTriple> events) {

        for (RelationTriple event1 : events) {
            for (RelationTriple event2 : events) {
                //System.out.println("\nevent1: "+formattedTriple(event1));
                //System.out.println("event2: "+formattedTriple(event2));
                if (event1.equals(event2)) {
                    //System.out.println("skipping");
                    continue;
                }

                //check if all the words in the shorter are in the longer
                //string; if so, eliminate the shorter string, and assign
                //event1 to be the longer string
                RelationTriple shorter = getShortAndLong(event1, event2).get(0);
                RelationTriple longer = getShortAndLong(event1, event2).get(1);
                if (longer.allTokens().containsAll(shorter.allTokens())) {
                    //System.out.println("longer string contains all words in shorter; eliminating shorter");
                    events.remove(shorter);
                    event1 = longer;
                } else {
                    //System.out.println("different strings; keep both");
                }
                //System.out.println("set now contains "+events.size()+" events");
                //System.out.println("event set is now: ");
                for (RelationTriple event : events) {
                    //System.out.println(formattedTriple(event));
                }
            }
        }
    }

    public static String formattedTriple(RelationTriple rt) {

        return rt.subjectGloss()+" "+rt.relationGloss()+" "+rt.objectGloss();
    }

    public static String renderFileAsString(Scanner fileIn) {
        ArrayList<String> fileLines = new ArrayList<String>();
        while(fileIn.hasNextLine()) {
            fileLines.add(fileIn.nextLine());
        }
        return String.join(" ",fileLines);
    }

    //method that takes in a list and removes all strings whose token length is less than n
    public static void removeShortStrings(Set<RelationTriple> events, int n) {

        for (RelationTriple rt : events) {
            if (formattedTriple(rt).split(" ").length < n) {
                events.remove(rt);
            }
        }
    }

    public static void extractEvents(String fileFolder, String resultsFile, StanfordCoreNLP pipeline,
                                     CRFClassifier model) throws Exception {


        PrintWriter fileOut = new PrintWriter(resultsFile);

        File folder = new File(fileFolder);
        File[] files = folder.listFiles();
        for (File file : files) {

            System.out.println("\nFILE: " + file.getName());
            fileOut.println("\nFILE: " + file.getName());
            Scanner fileIn = new Scanner(file);
            //String text = renderFileAsString(fileIn);

            //StanfordNER.doTagging(model, text);
            //fileOut.println();
            Set<RelationTriple> events = new CopyOnWriteArraySet<RelationTriple>();
            while (fileIn.hasNextLine()) {
                Annotation doc = new Annotation(fileIn.nextLine());
                pipeline.annotate(doc);

                for (CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class)) {
                    Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
                    for (RelationTriple triple : triples) {


                        String taggedSubject = StanfordNER.doTagging(model, triple.subjectGloss());
                        String taggedObject = StanfordNER.doTagging(model, triple.objectGloss());

                        ArrayList<String> tagListSubj = getTags(taggedSubject);
                        ArrayList<String> tagListObj = getTags(taggedObject);
                        if (hasNamedEntity(tagListSubj) || hasNamedEntity(tagListObj)) {
                            events.add(triple);
                        }
                    }
                }
            }

            StringLengthSorter sLenSorter = new StringLengthSorter();
            ArrayList<RelationTriple> eventsAsList = new ArrayList<>(events);
            Collections.sort(eventsAsList);
            //Collections.sort(eventsAsList, sLenSorter);
            /*
            fileOut.println("\nFull list of events in this file ("+eventsAsList.size()+")");
            for (RelationTriple event : eventsAsList) {
                fileOut.println(formattedTriple(event));
            }
            */

            eliminateShortenedEvents(events);
            removeShortStrings(events, 5);
            eventsAsList = new ArrayList<>(events);
            //Collections.sort(eventsAsList);
            Collections.sort(eventsAsList, sLenSorter);


            fileOut.println("\nEVENTS FOUND IN THIS FILE (" + eventsAsList.size() + ")\n");
            for (RelationTriple event : eventsAsList) {
                fileOut.println(formattedTriple(event));
            }
        }

        fileOut.close();
    }

    public static void main(String[]args) throws Exception {

        long startTime = System.currentTimeMillis();

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse,natlog,openie");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        CRFClassifier model = CRFClassifier.getClassifier(new File(args[0]));

        extractEvents(args[1], args[2], pipeline, model);
        extractEvents(args[3], args[4], pipeline, model);

        long endTime = System.currentTimeMillis();
        long seconds = (endTime - startTime) / 1000;
        long minutes = seconds / 60;
        seconds %= 60;
        System.out.println("\nRunning Time: "+minutes+" minutes "+seconds+" seconds");
    }
}










