import 'dart:developer';

import 'package:flutter/material.dart';
import 'package:langchain/langchain.dart';
import 'package:ollama/ollama.dart' as ola;
import 'package:langchain_ollama/langchain_ollama.dart';
import 'package:http/http.dart' as http;

final String IP_addr = "http://192.168.1.111";
final Ollama_endpt = '${IP_addr}:11434/api';
final ollama = ola.Ollama(baseUrl: Uri.parse(Ollama_endpt));
final deepSeekModel = "deepseek-r1:7b";

void main() {


  runApp(MaterialApp(home: MainApp()));
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});



  @override
  Widget build(BuildContext context) {
    final myController = TextEditingController();

    sendRAGmsg() async {

      final msg = myController.text;

      final String myData = "Dr. (Mr.) Chenshu Wu, HKU AIoT (Artificial Intelligence of Things) Lab Assistant Professor, Department of Computer Science Assistant Director, School of Computing and Data Science The University of Hong Kong Consultant, Origin AI Office: 315B, Chow Yei Ching Building Email: chenshu [at] cs [dot] hku [dot] hk";

      // 1. Create a vector store and add documents to it
      final vectorStore = MemoryVectorStore(
        embeddings: OllamaEmbeddings(model: "deepseek-r1:7b", baseUrl: Ollama_endpt),
      );
      await vectorStore.addDocuments(
        documents: [
          // Document(pageContent: 'LangChain was created by Harrison'),
          // Document(pageContent: 'David ported LangChain to Dart in LangChain.dart'),
          Document(pageContent: myData),
        ],
      );

      // 2. Define the retrieval chain
      final retriever = vectorStore.asRetriever();
      final setupAndRetrieval = Runnable.fromMap<String>({
        'context': retriever.pipe(
          Runnable.mapInput((docs) => docs.map((d) => d.pageContent).join('\n')),
        ),
        'question': Runnable.passthrough(),
      });

      // 3. Construct a RAG prompt template
      final promptTemplate = ChatPromptTemplate.fromTemplates([
        (ChatMessageType.system, 'Answer the question based on only the following context:\n{context}'),
        (ChatMessageType.human, '{question}'),
      ]);

      // 4. Define the final chain
      final model = ChatOllama(baseUrl: Ollama_endpt, defaultOptions: ChatOllamaOptions(model: "deepseek-r1:7b"));
      
      const outputParser = StringOutputParser<ChatResult>();
      final chain = setupAndRetrieval
          .pipe(promptTemplate)
          .pipe(model)
          .pipe(outputParser);

      // 5. Run the pipeline
      final res = await chain.invoke("${msg}");
      // print(res);

      // 6. filter the reasoning steps from DeepSeek model and show the response
      List<String> sp = res.split("</think>");
      // print(sp.last);

      var response_noThink = sp.last.trim();

      showDialog(
        context: context,
        builder: (context) {
        return AlertDialog(
            content: Text("Response:\n${response_noThink}"),
            scrollable: true,
          );
        },
      );

  }

  


    Future<void> sendMessage() async {
      final msg = myController.text;
      final stream = ollama.generate(
        msg,
        model: 'deepseek-r1:7b',
        chunked: false,
      );

      await for (final chunk in stream) {
        print(chunk);
        showDialog(
          context: context,
          builder: (context) {
          return AlertDialog(
              content: Text("Response:\n${chunk.toString()}"),
              scrollable: true,
            );
          },
        );
      }
    }

    return MaterialApp(
      title: 'DeepSeek RAG with Flutter',
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        body: Center(
          child: Padding(
            padding: EdgeInsets.all(20),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [   
                TextField(
                  keyboardType: TextInputType.multiline,
                  maxLines: null,
                  minLines: 1,
                  controller: myController,
                  decoration: InputDecoration(
                    border: OutlineInputBorder(),
                    hintText: 'Ask anything to DeepSeek',
                    labelText: 'Input',
                  ),
                ),

                const SizedBox(height: 30),
          
                ElevatedButton(
                  style: ElevatedButton.styleFrom(textStyle: const TextStyle(fontSize: 20)), 
                  onPressed: () async {
                    // sendMessage();
                    sendRAGmsg();
                  } , 
                  child: const Text('Send RAG')
                ),

                const SizedBox(height: 30),
          
                ElevatedButton(
                  style: ElevatedButton.styleFrom(textStyle: const TextStyle(fontSize: 20)), 
                  onPressed: () async {
                    // sendMessage();
                    sendMessage();
                  } , 
                  child: const Text('Send without RAG')
                ),
             
              ],
            )
          )
        ),
      ),
    );
  }

}
