// frontend/src/App.js
import React, { useState, useEffect } from 'react';

function App() {
  const [displayItems, setDisplayItems] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);

  useEffect(() => {
    fetch('/api/items')
      .then(res => res.json())
      .then(data => {
        // shuffle & pick 100
        const shuffled = [...data].sort(() => 0.5 - Math.random());
        setDisplayItems(shuffled.slice(0, 100));
      })
      .catch(console.error);
  }, []);

  const toggleSelect = id => {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  const handleSubmit = () => {
    console.log('Submitting:', selectedIds);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: 16,
      }}
    >
      {/* Header */}
      <div
        style={{
          marginBottom: 16,
          fontSize: 18,
          textAlign: 'center',
        }}
      >
        Selected: {selectedIds.length} (min 15)
        <button
          onClick={handleSubmit}
          disabled={selectedIds.length < 15}
          style={{ marginLeft: 16, padding: '8px 16px' }}
        >
          Submit
        </button>
      </div>

      {/* Grid container */}
      <div
        style={{
          width: '100%',
          maxWidth: '100vw',
          padding: '0 16px',
          boxSizing: 'border-box',
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
                  aspectRatio: '1 / 1',      // keep cells square
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

export default App;
