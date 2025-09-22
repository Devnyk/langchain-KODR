const dotenv = require("dotenv").config();
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { ChatMistralAI } = require("@langchain/mistralai");
const { HumanMessage, AIMessage } = require("@langchain/core/messages"); // for creating chat history
const { PromptTemplate } = require("@langchain/core/prompts");


// creating a model instance for Google Gemini

// const model = new ChatGoogleGenerativeAI({
//   model: "gemini-2.0-flash",
//   temperature: 0,
//   apiKey: process.env.GOOGLE_API_KEY,
// });
console.clear();

// creating a model instance for Mistral

const model = new ChatMistralAI({
  model: "mistral-large-latest",
  temperature: 0,
});

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

const chain = template.pipe(model);
chain
  .invoke({
    topic: "express",
    audience: "10 year old",
  })
  .then((response) => console.log(response.text));