package com.ecommerce.customer_chatbot;
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class Chatbot {

    // Simple QA pair
    static class QA {
        public String question;
        public String answer;
        public QA() {}
        public QA(String q, String a) {
            this.question = q;
            this.answer = a;
        }
    }

    private List<QA> dataset = new ArrayList<>();
    private Map<String, Double> idfMap = new HashMap<>();
    private List<Map<String, Double>> tfidfVectors = new ArrayList<>();

    // Common English stopwords
    private Set<String> stopwords = new HashSet<>(Arrays.asList(
            "a","an","the","is","am","are","was","were","be","been",
            "i","you","he","she","it","we","they","my","your","his","her","its","our","their",
            "on","in","at","to","for","of","and","or","but","how","what","where","when"
    ));

    public static void main(String[] args) throws IOException {
        Chatbot bot = new Chatbot();
        bot.loadDataset("Ecommerce_FAQ_Chatbot_dataset.json"); // your JSON file
        bot.preprocessDataset();
        bot.computeTFIDF();
        bot.chat();
    }

    // Load JSON dataset with root key "questions"
    void loadDataset(String fileName) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode rootNode = mapper.readTree(new File(fileName));
        JsonNode questionsNode = rootNode.get("questions"); // match your JSON key

        if (questionsNode != null && questionsNode.isArray()) {
            dataset = new ArrayList<>();
            for (JsonNode node : questionsNode) {
                String question = node.get("question").asText();
                String answer = node.get("answer").asText();
                dataset.add(new QA(question, answer));
            }
        }

        System.out.println("Loaded " + dataset.size() + " QA pairs.");
    }

    // Clean text: lowercase, remove punctuation, remove stopwords
 // Clean text: lowercase, remove punctuation, remove stopwords
    String cleanText(String text) {
        text = text.toLowerCase().replaceAll("[^a-z0-9\\s]", " ").trim();
        String[] words = text.split("\\s+");
        List<String> filtered = new ArrayList<>();
        for (String w : words) if (!stopwords.contains(w)) filtered.add(w);
        return String.join(" ", filtered);
    }

    // Generate unigrams + bigrams from cleaned text
    List<String> buildNgrams(String text) {
        String[] words = text.split("\\s+");
        List<String> ngrams = new ArrayList<>();
        // unigrams
        for(String w : words) ngrams.add(w);
        // bigrams
        for(int i=0; i<words.length-1; i++) {
            ngrams.add(words[i] + " " + words[i+1]);
        }
        return ngrams;
    }

    // Preprocess dataset questions
    void preprocessDataset() {
        for(QA qa : dataset) {
            qa.question = cleanText(qa.question);
        }
    }

    // Compute TF-IDF
    void computeTFIDF() {
        Map<String, Integer> df = new HashMap<>();
        List<Map<String, Integer>> tfList = new ArrayList<>();

        for(QA qa : dataset) {
            Map<String, Integer> tf = new HashMap<>();
            List<String> ngrams = buildNgrams(qa.question);
            for(String term : ngrams) tf.put(term, tf.getOrDefault(term,0)+1);
            tfList.add(tf);

            Set<String> uniqueTerms = new HashSet<>(ngrams);
            for(String term : uniqueTerms) df.put(term, df.getOrDefault(term,0)+1);
        }


        int totalDocs = dataset.size();
        for(String term : df.keySet()) {
            idfMap.put(term, Math.log((double) totalDocs / (df.get(term) + 1)));
        }

        for(Map<String, Integer> tf : tfList) {
            Map<String, Double> tfidf = new HashMap<>();
            for(String term : tf.keySet()) {
                tfidf.put(term, tf.get(term) * idfMap.getOrDefault(term,0.0));
            }
            tfidfVectors.add(tfidf);
        }

        System.out.println("TF-IDF computed.");
    }

    // Build TF-IDF vector for user input
    Map<String, Double> buildVector(String text) {
        String cleaned = cleanText(text);
        String[] words = cleaned.split("\\s+");
        Map<String, Integer> tf = new HashMap<>();
        for(String w : words) tf.put(w, tf.getOrDefault(w,0)+1);

        Map<String, Double> vector = new HashMap<>();
        for(String term : tf.keySet()) {
            vector.put(term, tf.get(term) * idfMap.getOrDefault(term,0.0));
        }
        return vector;
    }

    // Cosine similarity
    double cosineSimilarity(Map<String, Double> v1, Map<String, Double> v2) {
        Set<String> terms = new HashSet<>();
        terms.addAll(v1.keySet());
        terms.addAll(v2.keySet());
        double dot = 0, norm1 = 0, norm2 = 0;
        for(String t : terms) {
            double a = v1.getOrDefault(t,0.0);
            double b = v2.getOrDefault(t,0.0);
            dot += a*b;
            norm1 += a*a;
            norm2 += b*b;
        }
        if(norm1==0 || norm2==0) return 0;
        return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    // Chat loop
    void chat() throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        System.out.println("TF-IDF Chatbot (type 'exit' to quit)");
        while(true) {
            System.out.print("You: ");
            String input = br.readLine();
            if(input.equalsIgnoreCase("exit")) break;

            Map<String, Double> inputVec = buildVector(input);
            double maxSim = -1;
            int bestIndex = -1;
            for(int i=0;i<tfidfVectors.size();i++) {
                double sim = cosineSimilarity(inputVec, tfidfVectors.get(i));
                if(sim > maxSim) {
                    maxSim = sim;
                    bestIndex = i;
                }
            }

            if(maxSim > 0.01) { // small threshold
                System.out.println("Bot: " + dataset.get(bestIndex).answer);
            } else {
                System.out.println("Bot: Sorry, I didn't understand that.");
            }
        }
    }
}
