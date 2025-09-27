require("dotenv").config();
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { Pinecone: PineconeClient } = require("@pinecone-database/pinecone");

const emmbedder = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
  apiKey: process.env.GOOGLE_API_KEY,
});

const pineconeIndex = new PineconeClient({
  apiKey: process.env.PINECONE_API_KEY,
}).Index(process.env.PINECONE_INDEX);

async function queryPinecone(text, topK = 3) {
  const vector = await emmbedder.embedQuery(text);
  return pineconeIndex.query({ vector, topK, includeMetadata: true });
}

(async () => {
  try {
    const result = await queryPinecone("express?");
    console.log(result);
  } catch (err) {
    console.error("Error:", err.message);
  }
})();
