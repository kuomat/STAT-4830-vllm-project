// frontend/src/ResultsPage.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function ResultsPage({ selectedIds, recommendations, isPersonaSelection }) {
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
        gap: 10,
      }}
    >
      {ids.map(id => {
        const item = itemsMap[id];
        return (
          <div
            key={id}
            onClick={() => item && setModalItem(item)}
            style={{
              cursor: 'pointer',
              border: '1px solid #e0e0e0',
              borderRadius: '4px',
              overflow: 'hidden'
            }}
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
    <div style={{
      padding: 24,
      textAlign: 'center',
      maxWidth: '1200px',
      margin: '0 auto',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      {/* Back button */}
      <button
        onClick={() => navigate('/')}
        style={{
          marginBottom: 24,
          padding: '8px 16px',
          backgroundColor: '#3498db',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
      >
        ← Select Again
      </button>

      {/* Selections: 10 across - Only show if NOT a persona selection */}
      {!isPersonaSelection && (
        <>
          <h2 style={{
            marginBottom: '20px',
            color: '#2c3e50'
          }}>Your Selections</h2>
          {selectedIds && selectedIds.length > 0 ? (
            renderGrid(selectedIds, 10)
          ) : (
            <p style={{ color: '#7f8c8d' }}>No items selected.</p>
          )}
        </>
      )}

      {/* Only show dropdown if there's at least one algo */}
      {algoKeys.length > 0 && (
        <div style={{ marginTop: isPersonaSelection ? 0 : 32, marginBottom: 16 }}>
          <label htmlFor="algo-select" style={{ marginRight: 12, fontWeight: 'bold' }}>
            View recommendations from:
          </label>
          <select
            id="algo-select"
            value={currentAlgo}
            onChange={(e) => setCurrentAlgo(e.target.value)}
            style={{ padding: '6px 12px', borderRadius: '4px' }}
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
          <h3 style={{
            marginBottom: '16px',
            color: '#2c3e50'
          }}>{currentAlgo.replace('-', ' ').toUpperCase()}</h3>
          {recommendations[currentAlgo].length > 0
            ? renderGrid(recommendations[currentAlgo], 5)
            : <p style={{ color: '#7f8c8d' }}>No recommendations.</p>
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
              width: '90%',
              textAlign: 'left',
              maxHeight: '80vh',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <h2 style={{ margin: '0 0 8px 0', color: '#2c3e50' }}>{modalItem.name}</h2>
            <p style={{ margin: '0 0 16px 0', fontSize: '18px' }}>
              <strong style={{ color: '#3498db' }}>${modalItem.price.toFixed(2)}</strong>
            </p>
            <div style={{
              overflowY: 'auto',
              flex: '1 1 auto',
            }}>
              <p style={{
                whiteSpace: 'pre-wrap',
                margin: 0,
                lineHeight: '1.5',
                color: '#333'
              }}>
                {modalItem.description}
              </p>
            </div>
            <button
              onClick={() => setModalItem(null)}
              style={{
                marginTop: 16,
                padding: '8px 0',
                backgroundColor: '#3498db',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
