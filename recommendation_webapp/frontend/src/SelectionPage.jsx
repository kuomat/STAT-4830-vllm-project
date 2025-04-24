// frontend/src/SelectionPage.jsx
import React, { useState, useEffect, useMemo, useCallback, memo } from 'react';

// Import persona data directly
import personaRatings from './persona_ratings.json';
import personaDescriptions from './persona_descriptions.json';

// Memoized item component for better performance
const GridItem = memo(({ item, isSelected, onToggle }) => (
  <div
    onClick={() => onToggle(item.id)}
    style={{
      width: '100%',
      aspectRatio: '1 / 1',
      border: isSelected
        ? '3px solid #2ecc71'
        : '1px solid #dfe6e9',
      borderRadius: '8px',
      cursor: 'pointer',
      overflow: 'hidden',
      willChange: 'transform, box-shadow',
      transform: isSelected ? 'scale(1.05)' : 'scale(1)',
      boxShadow: isSelected ? '0 4px 8px rgba(0,0,0,0.15)' : '0 2px 4px rgba(0,0,0,0.05)',
      position: 'relative'
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
      loading="lazy"
    />
    {isSelected && (
      <div style={{
        position: 'absolute',
        top: '5px',
        right: '5px',
        backgroundColor: '#2ecc71',
        borderRadius: '50%',
        width: '20px',
        height: '20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        fontSize: '12px',
        fontWeight: 'bold'
      }}>
        ✓
      </div>
    )}
  </div>
));

export default function SelectionPage({ onSubmit }) {
  const [displayItems, setDisplayItems] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [selectedPersona, setSelectedPersona] = useState(null);
  const [hoverPersona, setHoverPersona] = useState(null);

  //// For example personas ////
  const ex1_name = 'Riley';
  const ex2_name = 'Emerson';
  const ex3_name = 'Noah';

  const ex1_ratings = personaRatings[ex1_name];
  const ex2_ratings = personaRatings[ex2_name];
  const ex3_ratings = personaRatings[ex3_name];

  useEffect(() => {
    fetch('/api/items')
      .then(res => res.json())
      .then(data => {
        // Only shuffle once and limit to 100 items
        const shuffled = [...data].sort(() => 0.5 - Math.random());
        setDisplayItems(shuffled.slice(0, 100));
      })
      .catch(console.error);
  }, []);

  // Memoize selected IDs set for faster lookups
  const selectedIdsSet = useMemo(() => {
    const set = new Set();
    selectedIds.forEach(id => set.add(id));
    return set;
  }, [selectedIds]);

  // Use callback to prevent recreation on every render
  const toggleSelect = useCallback(id => {
    setSelectedIds(prev =>
      prev.includes(id)
        ? prev.filter(x => x !== id)
        : [...prev, id]
    );
  }, []);

  const handlePersonaSelect = useCallback((personaName, ratings) => {
    // Convert ratings to an array of ids rated highly (e.g., rating > 3)
    const selectedRatings = Object.entries(ratings)
      .filter(([_, rating]) => rating > 3)
      .map(([id, _]) => id);

    // Set the selected persona for display
    setSelectedPersona({
      name: personaName,
      description: personaDescriptions[personaName]
    });

    // Submit these ratings as if the user selected them, but with isPersona=true
    onSubmit(selectedRatings, 10, true);
  }, [onSubmit]);

  const handlePersonaHover = useCallback((personaName) => {
    if (personaName) {
      setHoverPersona({
        name: personaName,
        description: personaDescriptions[personaName]
      });
    } else {
      setHoverPersona(null);
    }
  }, []);

  const buttonStyle = {
    backgroundColor: '#3498db',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    padding: '10px 20px',
    fontSize: '16px',
    fontWeight: '600',
    margin: '0 10px 20px 10px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
    position: 'relative'
  };

  // Memoize the grid items to prevent unnecessary re-renders
  const gridItems = useMemo(() => {
    return displayItems.map(item => {
      const isSelected = selectedIdsSet.has(item.id);
      return (
        <GridItem
          key={item.id}
          item={item}
          isSelected={isSelected}
          onToggle={toggleSelect}
        />
      );
    });
  }, [displayItems, selectedIdsSet, toggleSelect]);

  return (
    <div style={{
      padding: '24px 32px',
      textAlign: 'center',
      backgroundColor: '#f8f9fa',
      minHeight: '100vh',
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <h1 style={{
        color: '#2c3e50',
        marginBottom: '24px',
        fontSize: '28px'
      }}>
        Fashion Recommendation System
      </h1>

      {/* Persona Buttons */}
      <div style={{
        marginBottom: '32px',
        padding: '20px',
        backgroundColor: 'white',
        borderRadius: '12px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.08)'
      }}>
        <h3 style={{
          marginBottom: '16px',
          color: '#2c3e50',
          fontSize: '18px'
        }}>
          Try with example personas:
        </h3>
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          flexWrap: 'wrap',
          position: 'relative'
        }}>
          <button
            style={buttonStyle}
            onClick={() => handlePersonaSelect(ex1_name, ex1_ratings)}
            onMouseEnter={() => handlePersonaHover(ex1_name)}
            onMouseLeave={() => handlePersonaHover(null)}
          >
            {ex1_name}
          </button>
          <button
            style={buttonStyle}
            onClick={() => handlePersonaSelect(ex2_name, ex2_ratings)}
            onMouseEnter={() => handlePersonaHover(ex2_name)}
            onMouseLeave={() => handlePersonaHover(null)}
          >
            {ex2_name}
          </button>
          <button
            style={buttonStyle}
            onClick={() => handlePersonaSelect(ex3_name, ex3_ratings)}
            onMouseEnter={() => handlePersonaHover(ex3_name)}
            onMouseLeave={() => handlePersonaHover(null)}
          >
            {ex3_name}
          </button>
        </div>

        {hoverPersona && (
          <div style={{
            marginTop: '10px',
            padding: '16px',
            backgroundColor: '#f1f8ff',
            borderRadius: '8px',
            textAlign: 'left',
            transition: 'opacity 0.3s ease',
            opacity: 1
          }}>
            <h4 style={{
              marginBottom: '10px',
              color: '#2c3e50'
            }}>
              {hoverPersona.name}'s Style Profile:
            </h4>
            <p style={{
              lineHeight: '1.6',
              color: '#34495e'
            }}>
              {hoverPersona.description}
            </p>
          </div>
        )}

        {selectedPersona && !hoverPersona && (
          <div style={{
            marginTop: '20px',
            padding: '16px',
            backgroundColor: '#e8f5e9',
            borderRadius: '8px',
            textAlign: 'left',
            border: '1px solid #c8e6c9'
          }}>
            <h4 style={{
              marginBottom: '10px',
              color: '#2c3e50'
            }}>
              Selected: {selectedPersona.name}'s Style Profile
            </h4>
            <p style={{
              lineHeight: '1.6',
              color: '#34495e'
            }}>
              {selectedPersona.description}
            </p>
          </div>
        )}
      </div>

      {/* Select Items Section */}
      <h2 style={{
        color: '#2c3e50',
        marginBottom: '20px',
        fontSize: '22px'
      }}>
        Select Items
      </h2>

      <div style={{
        marginBottom: '24px',
        padding: '16px',
        backgroundColor: 'white',
        borderRadius: '12px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{
          fontSize: '18px',
          fontWeight: '600',
          color: '#2c3e50'
        }}>
          Selected: <span style={{ color: '#3498db' }}>{selectedIds.length}</span>
          <span style={{
            fontSize: '14px',
            color: '#7f8c8d',
            marginLeft: '4px'
          }}>
            (min 15)
          </span>
        </div>
        <button
          onClick={() => onSubmit(selectedIds)}
          disabled={selectedIds.length < 15}
          style={{
            backgroundColor: selectedIds.length < 15 ? '#cbd5e0' : '#3498db',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '10px 24px',
            fontSize: '16px',
            fontWeight: '600',
            cursor: selectedIds.length < 15 ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease',
            boxShadow: '0 2px 5px rgba(0,0,0,0.1)'
          }}
        >
          Submit
        </button>
      </div>

      <div
        style={{
          width: '100%',
          maxWidth: '1200px',
          margin: '0 auto',
        }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
            gap: '12px',
          }}
        >
          {gridItems}
        </div>
      </div>
    </div>
  );
}
