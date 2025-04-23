// frontend/src/ResultsPage.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function ResultsPage({ selectedIds, recommendations }) {
  const navigate = useNavigate();
  const algoKeys = Object.keys(recommendations);
  const [currentAlgo, setCurrentAlgo] = useState(algoKeys[0] || '');
  const [itemsMap, setItemsMap] = useState({});
  const [modalItem, setModalItem] = useState(null);

  // if recommendations change (e.g. when you add more methods), reset dropdown
  useEffect(() => {
    if (algoKeys.length) setCurrentAlgo(algoKeys[0]);
  }, [recommendations]);

  useEffect(() => {
    fetch('/api/items')
      .then(r => r.json())
      .then(data => {
        const map = {};
        data.forEach(item => {
          map[item.id] = item;
        });
        setItemsMap(map);
      })
      .catch(console.error);
  }, []);


  // generic grid renderer with dynamic columns
  const renderGrid = (ids, columns) => (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: 8,
      }}
    >
    {ids.map(id => {
      const item = itemsMap[id];
      return (
        <div
          key={id}
          onClick={() => item && setModalItem(item)}
          style={{ cursor: 'pointer' }}
        >
          <img
            src={`/images/${id}.png`}
            alt={item?.name || id}
            style={{ width: '100%', objectFit: 'cover' }}
          />
        </div>
      );
    })}
    </div>
  );


  return (
    <div style={{ padding: 16, textAlign: 'center' }}>
      {/* Back button */}
      <button
        onClick={() => navigate('/')}
        style={{ marginBottom: 24, padding: '8px 16px' }}
      >
        ← Select Again
      </button>

      {/* Selections: 10 across */}
      <h2>Your Selections</h2>
      {selectedIds && selectedIds.length > 0 ? (
        renderGrid(selectedIds, 10)
      ) : (
        <p>No items selected.</p>
      )}

      {/* Only show dropdown if there’s at least one algo */}
      {algoKeys.length > 0 && (
        <div style={{ marginTop: 32 }}>
          <label htmlFor="algo-select" style={{ marginRight: 8 }}>
            View recommendations from:
          </label>
          <select
            id="algo-select"
            value={currentAlgo}
            onChange={(e) => setCurrentAlgo(e.target.value)}
            style={{ padding: '4px 8px' }}
          >
            {algoKeys.map((key) => (
              <option key={key} value={key}>
                {key.replace('-', ' ').toUpperCase()}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Recommendations: 5 across */}
      {currentAlgo && Array.isArray(recommendations[currentAlgo]) && (
        <div style={{ marginTop: 24 }}>
          <h3>{currentAlgo.replace('-', ' ').toUpperCase()}</h3>
          {recommendations[currentAlgo].length > 0
            ? renderGrid(recommendations[currentAlgo], 5)
            : <p>No recommendations.</p>
          }
        </div>
      )}

      {modalItem && (
              <div
                style={{
                  position: 'fixed',
                  top: 0, left: 0, right: 0, bottom: 0,
                  background: 'rgba(0,0,0,0.6)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  zIndex: 1000,
                }}
                onClick={() => setModalItem(null)}
              >
                <div
                  onClick={e => e.stopPropagation()}
                  style={{
                    background: '#fff',
                    padding: 24,
                    borderRadius: 8,
                    maxWidth: 400,
                    textAlign: 'left',
                  }}
                >
                  <h2>{modalItem.name}</h2>
                  <p><strong>${modalItem.price.toFixed(2)}</strong></p>
                  <p style={{ whiteSpace: 'pre-wrap', maxHeight: '60vh', overflowY: 'auto' }}>
                    {modalItem.description}
                  </p>
                  <button onClick={() => setModalItem(null)} style={{ marginTop: 16 }}>
                    Close
                  </button>
                </div>
              </div>
            )}









    </div>
  );
}
