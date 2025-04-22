// frontend/src/SelectionPage.jsx
import React, { useState, useEffect } from 'react';

export default function SelectionPage({ onSubmit }) {
  const [displayItems, setDisplayItems] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);

  useEffect(() => {
    fetch('/api/items')
      .then(res => res.json())
      .then(data => {
        const shuffled = [...data].sort(() => 0.5 - Math.random());
        setDisplayItems(shuffled.slice(0, 100));
      })
      .catch(console.error);
  }, []);

  const toggleSelect = id => {
    setSelectedIds(prev =>
      prev.includes(id)
        ? prev.filter(x => x !== id)
        : [...prev, id]
    );
  };

  return (
    <div style={{ padding: 16, textAlign: 'center' }}>
      <div style={{ marginBottom: 16, fontSize: 18 }}>
        Selected: {selectedIds.length} (min 15)
        <button
          onClick={() => onSubmit(selectedIds)}
          disabled={selectedIds.length < 15}
          style={{ marginLeft: 16, padding: '8px 16px' }}
        >
          Submit
        </button>
      </div>
      <div
        style={{
          width: '100%',
          maxWidth: '100vw',
          padding: '0 16px',
          boxSizing: 'border-box',
          margin: '0 auto',
        }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(10, 1fr)',
            gap: 8,
          }}
        >
          {displayItems.map(item => {
            const isSelected = selectedIds.includes(item.id);
            return (
              <div
                key={item.id}
                onClick={() => toggleSelect(item.id)}
                style={{
                  width: '100%',
                  aspectRatio: '1 / 1',
                  border: isSelected
                    ? '3px solid #4caf50'
                    : '1px solid #ccc',
                  borderRadius: 4,
                  cursor: 'pointer',
                  overflow: 'hidden',
                }}
              >
                <img
                  src={item.image_url}
                  alt={item.id}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                  }}
                />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
