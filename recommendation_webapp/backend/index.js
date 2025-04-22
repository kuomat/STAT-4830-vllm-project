// backend/index.js
const express = require('express');
const cors = require('cors');
const items = require('./data/items.json');
const fetch = require('node-fetch');

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

// … in your routes section …
app.get('/api/items', (req, res) => {
  res.json(items);
});

app.post('/api/recommend/:method', async (req, res) => {
    const { method } = req.params;
    const { selectedIds, n } = req.body;
    try {
      const pyRes = await fetch(`http://localhost:8000/recommend/${method}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selected: selectedIds, n }),
      });
      const json = await pyRes.json();
      res.json(json);
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: 'Recommendation service error' });
    }
  });
  