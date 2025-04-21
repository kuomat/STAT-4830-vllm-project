// backend/index.js
const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Example endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK' });
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Backend listening on http://localhost:${PORT}`);
});

const path = require('path');
// serve everything in /public/images at /images/*
app.use(
  '/images',
  express.static(path.join(__dirname, 'public/images/dataset'))
);

// near top of file
const items = require('./data/items.json');

// … in your routes section …
app.get('/api/items', (req, res) => {
  res.json(items);
});
