const dotenv = require("dotenv").config();
const {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} = require("@langchain/google-genai");
const { ChatMistralAI } = require("@langchain/mistralai");
const {
  HumanMessage,
  AIMessage,
  ToolMessage,
} = require("@langchain/core/messages"); // for creating chat history
const { PromptTemplate } = require("@langchain/core/prompts");
const { tool } = require("@langchain/core/tools");
const { z, Schema } = require("zod");
const { StateGraph, MessagesAnnotation } = require("@langchain/langgraph");
const { tavily } = require("@tavily/core");
const { PineconeStore } = require("@langchain/pinecone");
const { Pinecone: PineconeClient } = require("@pinecone-database/pinecone");
// creating a model instance for Google Gemini

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0,
  apiKey: process.env.GOOGLE_API_KEY,
});
console.clear();

// creating a model instance for Mistral

// const model = new ChatMistralAI({
//   model: "mistral-large-latest",
//   temperature: 0,
// });

// generating a simple response

// model.invoke("write haiku about the sea").then((response) => console.log(response.text));

// generating a streaming response

// model.stream("write haiku about the sea").then(async (response) => {
//   for await (const chunk of response) {
//     console.log(chunk.text);
//   }
// });

// creating a chat history

// const chatHistroy = [
//   new HumanMessage("hey"),
//   new AIMessage("hello"),
//   new HumanMessage("what was my first question?"),
// ];

// creating a prompt template

const template = PromptTemplate.fromTemplate(
  `explain {topic} in a simple way for a {audience} to understand.`
);

// using the prompt template with the model

// template.invoke({
//     topic:"express",
//     audience:"10 year old"
// }).then(temp =>model.invoke(temp).then((response) => console.log(response.text)))

// using pipe to combine prompt template and model together

// const chain = template.pipe(model);
// chain
//   .invoke({
//     topic: "express",
//     audience: "10 year old",
//   })
//   .then((response) => console.log(response.text));

// creating a tool for the model to add two numbers
// const addTwoNumbers = tool(
//   async ({ a, b }) => {
//     return `the sum of ${a} and ${b} is ${a + b}`;
//   },
//   {
//     name: "addTwoNumbers",
//     description: "adds two numbers together and returns the result",
//     schema: z.object({
//       a: z.number().describe("the first number to add"),
//       b: z.number().describe("the second number to add"),
//     }),
//   }
// );

// binding the tool to the model
// const modelWithTool = model.bindTools([addTwoNumbers]);
// invoking the model with the tool
// the model will decide when to use the tool based on the user input
// here we are asking the model to add two numbers
// modelWithTool.invoke([new HumanMessage("what is 2 + 3?")]).then((response) => {
//   const callToTool = response.tool_calls[0];
//   addTwoNumbers.invoke(callToTool.args).then((toolResponse) => {
//     console.log("Response from tool: ", toolResponse);
//   });
// });

// using tavily to search the web example

// async function main() {
//   const tvly = tavily({ apiKey: process.env.TAVIT_API_KEY });
//   const response = await tvly.search("Who is Leo Messi?");
//   console.log(response);
// }

// main();

// const searchTool = tool(
//   async ({ query = "" }) => {
//     const tavilySearch = tavily({ apiKey: process.env.TAVILY_API_KEY });
//     const result = await tavilySearch.search(query);
//     return JSON.stringify(result.results);
//   },
//   {
//     name: "searchTool",
//     description: "useful for when you need to answer questions about topics.",
//     schema: z.object({
//       query: z.string().describe("the search query"),
//     }),
//   }
// );

// const graph = new StateGraph(MessagesAnnotation)
//   .addNode("LLM", async (state) => {
//     const lastMessage = state.messages[state.messages.length - 1];
//     const modelWithBindTool = model.bindTools([searchTool]);
//     const response = await modelWithBindTool.invoke([lastMessage]);
//     state.messages.push(response);
//     return state;
//   })
//   .addNode("TOOLS", async (state) => {
//     const lastMessage = state.messages[state.messages.length - 1];
//     const toolCall = lastMessage.tool_calls?.[0];
//     const result = await searchTool.invoke(toolCall?.args ?? {});
//     const toolMessage = new ToolMessage({
//       name: toolCall?.name,
//       content: result,
//     });
//     state.messages.push(toolMessage);
//     return state;
//   })
//   .addEdge("__start__", "LLM")
//   .addEdge("TOOLS", "LLM")
//   .addConditionalEdges("LLM", async (state) => {
//     const lastMessage = state.messages[state.messages.length - 1];
//     if (lastMessage.tool_calls?.length) {
//       return "TOOLS";
//     }
//     return "__end__";
//   });

// const agent = graph.compile(); // this creates the instance of model + tool graph for use

// using the above graph instacne
// agent
//   .invoke(
//     {
//       messages: [new HumanMessage("Who is Leo Messi?")],
//     },
//     { recursionLimit: 5 }
//   )
//   .then((response) => {
//     console.log(
//       "Final response: ",
//       response.messages[response.messages.length - 1].text
//     );
//   });

// model instance generating embeddigns in 3071D
// const emmbedder = new GoogleGenerativeAIEmbeddings({
//   model: "gemini-embedding-001",
//   apiKey: process.env.GOOGLE_API_KEY,
// });

// querying into model to generate embeddings of the txt
// emmbedder
//   .embedQuery("hii")
//   .then((e) => {
//     console.log(e);
//   })
//   .catch((err) => console.log(err.message));

// Vector database {PineCone}

// generating 768D embedding matching the index size using different model
const emmbedder = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
  apiKey: process.env.GOOGLE_API_KEY,
});

//configuring pinecone

const pinecone = new PineconeClient({
  apiKey: process.env.PINECONE_API_KEY,
});

const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

// generating vectors ans storting data to pinecone
// emmbedder
//   .embedQuery(
//     `Express is a fast, unopinionated, and minimalist web framework for Node.js, designed for building web applications and APIs.
//  It acts as a layer between the core Node.js server and an application's logic, simplifying the management of routes, requests, and responses.
//  Often described as the de facto standard server framework for Node.js, Express is built on top of Node.js and provides a robust set of features for handling HTTP requests, routing, middleware, and more.

// Its core philosophy emphasizes minimalism and flexibility, offering a lightweight foundation that can be easily augmented with middleware modules to add functionality like authentication, logging, request parsing, and error handling.
//  Express's powerful routing system allows developers to define URL patterns and HTTP methods (such as GET, POST, PUT, DELETE) in a clean and organized manner, making it particularly well-suited for creating RESTful APIs.
//  The framework supports integration with various templating engines (like Pug, EJS, and Handlebars) for rendering dynamic HTML pages directly from the server.`
//   )
//   .then(async (vectors) => {
//     await pineconeIndex.upsert([
//       {
//         id: "express-data",
//         values: vectors,
//         metadata: { genere: "action" },
//       },
//     ]);
//   })
//   .catch((err) => console.log(err.message));

// Query in pinecone
emmbedder.embedQuery("express?").then(async (vector) => {
  const result = await pineconeIndex.query({
    vector: vector,
    topK: 3,
    includeMetadata:true
  });
  console.log(result);
});