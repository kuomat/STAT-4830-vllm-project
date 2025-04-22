// frontend/src/App.js
import React, { useState } from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import SelectionPage from './SelectionPage';
import ResultsPage   from './ResultsPage';

function App() {
  const [selectedIds, setSelectedIds]       = useState([]);
  const [recommendations, setRecommendations] = useState({});
  const navigate = useNavigate();

  const onSubmit = async (ids, n = 10) => {
    setSelectedIds(ids);
    const methods = ['content-filtering', 'collaborative-filtering', 'low-rank']; // add your other methods later
    const results = await Promise.all(
      methods.map(m =>
        fetch(`/api/recommend/${m}`, {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ selectedIds: ids, n })
        }).then(r => r.json())
      )
    );

    const recs = {};
    methods.forEach((m, i) => recs[m] = results[i].recommendations);
    setRecommendations(recs);

    // use an absolute path here
    navigate('/results');
  };

  return (
    <Routes>
      {/* explicit “/” route for your selection page */}
      <Route path="/" element={<SelectionPage onSubmit={onSubmit} />} />
      {/* explicit “/results” route for the results page */}
      <Route
        path="/results"
        element={
          <ResultsPage
            selectedIds={selectedIds}
            recommendations={recommendations}
          />
        }
      />
      {/* optional catch‐all to redirect back to “/” */}
      <Route path="*" element={<SelectionPage onSubmit={onSubmit} />} />
    </Routes>
  );
}

export default App;
