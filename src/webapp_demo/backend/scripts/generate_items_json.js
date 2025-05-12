// backend/scripts/generate_items_json.js

const csv = require('csvtojson');
const fs = require('fs');
const path = require('path');

const CSV_PATH = path.join(__dirname, '../../../../dataset/embeddings_final.csv');
const OUT_PATH = path.join(__dirname, '../data/items.json');

csv()
  .fromFile(CSV_PATH)
  .then(rows => {
    const items = [];

    rows.forEach((row, idx) => {
      // Skip if any required field is missing or empty
      if (
        !row.name ||
        !row.description ||
        !row.price ||
        !row.image_key ||
        !row.text_embedding ||
        !row.image_embedding ||
        !row.image_path
      ) {
        console.warn(`Skipping row ${idx}: missing field`);
        return;
      }

      // Parse price
      const price = parseFloat(row.price);
      if (Number.isNaN(price)) {
        console.warn(`Skipping row ${idx}: invalid price`);
        return;
      }

      // Parse embeddings, skip on error
      let text_embedding, image_embedding;
      try {
        text_embedding = JSON.parse(row.text_embedding);
      } catch {
        console.warn(`Skipping row ${idx}: invalid text_embedding`);
        return;
      }
      try {
        image_embedding = JSON.parse(row.image_embedding);
      } catch {
        console.warn(`Skipping row ${idx}: invalid image_embedding`);
        return;
      }

      // Derive filename from image_path (e.g. "./dataset/2.png" → "2.png")
      const filename = path.basename(row.image_path);

      items.push({
        id: row.image_key,
        title: row.name,
        description: row.description,
        price,
        image_url: `/images/${filename}`,
        text_embedding,
        image_embedding,
      });
    });

    fs.writeFileSync(OUT_PATH, JSON.stringify(items, null, 2));
    console.log(`→ Wrote ${items.length} items to ${OUT_PATH}`);
  })
  .catch(err => {
    console.error('Error generating items.json:', err);
    process.exit(1);
  });
