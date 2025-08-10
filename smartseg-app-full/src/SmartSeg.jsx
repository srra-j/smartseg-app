/*
SmartSeg - Customer Segmentation App (single-file React component)

How to use:
1. Create a React project (Vite or Create React App) with Tailwind CSS set up.
2. Install dependencies:
   npm install papaparse recharts framer-motion

3. Place this file as `SmartSeg.jsx` in your src/ directory and import it in App.jsx:
   import SmartSeg from './SmartSeg';

   function App(){
     return <SmartSeg />;
   }

Notes:
- This file implements CSV upload (or generates a sample dataset), data standardization, a simple K-Means implementation in JS, PCA for 2D visualization, interactive UI to pick features and number of clusters, cluster profiling, and CSV download of labeled data.
- Styling uses Tailwind utility classes. Tailwind should be configured in your app for the UI to look as intended.
*/

import React, { useState, useMemo, useRef } from 'react';
import Papa from 'papaparse';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';

// ---------- Helper math utilities ----------
function mean(arr){
  if(arr.length===0) return 0;
  return arr.reduce((a,b)=>a+b,0)/arr.length;
}
function std(arr){
  const m = mean(arr);
  const v = mean(arr.map(x=> (x-m)*(x-m)));
  return Math.sqrt(v);
}

function transpose(matrix){
  return matrix[0].map((_,i)=>matrix.map(row=>row[i]));
}

// Simple PCA using covariance matrix and power iteration for top 2 eigenvectors
function pca2(data){
  // data: array of rows, numeric
  const n = data.length;
  if(n===0) return {scores:[], explained:[]};
  const m = data[0].length;
  // center
  const colMeans = [];
  for(let j=0;j<m;j++) colMeans.push(mean(data.map(r=>r[j])));
  const centered = data.map(r=>r.map((v,j)=>v-colMeans[j]));
  // covariance matrix (m x m)
  const cov = Array.from({length:m},()=>Array(m).fill(0));
  for(let i=0;i<n;i++){
    for(let a=0;a<m;a++){
      for(let b=0;b<m;b++){
        cov[a][b] += centered[i][a]*centered[i][b];
      }
    }
  }
  for(let a=0;a<m;a++) for(let b=0;b<m;b++) cov[a][b] /= (n-1||1);

  // Helper: multiply matrix x vector
  function mv(mat, vec){
    return mat.map(row=>row.reduce((s, val, i)=> s + val*vec[i], 0));
  }
  function norm(vec){
    return Math.sqrt(vec.reduce((s,v)=>s+v*v,0));
  }
  // power iteration to find top eigenvector
  function topEigen(mat){
    let v = Array(mat.length).fill(0).map(()=>Math.random());
    for(let it=0; it<200; it++){
      const w = mv(mat, v);
      const nrm = norm(w)||1;
      v = w.map(x=>x/nrm);
    }
    const lambda = mv(mat, v).reduce((s,x,i)=> s + x*v[i], 0);
    return {eig:lambda, vec:v};
  }
  const e1 = topEigen(cov);
  // deflate
  const outer = (a,b)=> a.map(x=> b.map(y=> x*y));
  const def = cov.map((row,i)=> row.map((val,j)=> val - e1.eig * e1.vec[i]*e1.vec[j]));
  const e2 = topEigen(def);
  const components = [e1.vec, e2.vec]; // m-length each
  // project
  const scores = centered.map(row=>({
    x: row.reduce((s,val,i)=> s + val*components[0][i],0),
    y: row.reduce((s,val,i)=> s + val*components[1][i],0)
  }));
  const totalVar = e1.eig + e2.eig + 1e-9;
  const explained = [e1.eig/totalVar, e2.eig/totalVar];
  return {scores, explained};
}

// K-Means implementation
function euclidean(a,b){
  let s=0; for(let i=0;i<a.length;i++){ const d=a[i]-b[i]; s+=d*d; } return Math.sqrt(s);
}

function kmeans(data, k, maxIter=100){
  // data: array of numeric arrays
  const n = data.length;
  if(n===0) return {labels:[], centroids:[]};
  const dim = data[0].length;
  // init centroids: pick k random distinct points
  const centroids = [];
  const used = new Set();
  const rng = ()=> Math.floor(Math.random()*n);
  while(centroids.length < k){
    const idx = rng();
    if(!used.has(idx)) { used.add(idx); centroids.push([...data[idx]]); }
  }
  let labels = Array(n).fill(0);
  for(let iter=0; iter<maxIter; iter++){
    let moved = false;
    // assign
    for(let i=0;i<n;i++){
      let best=0; let bestd=Infinity;
      for(let c=0;c<k;c++){
        const d = euclidean(data[i], centroids[c]);
        if(d<bestd){ bestd=d; best=c; }
      }
      if(labels[i] !== best){ moved = true; labels[i] = best; }
    }
    // recompute centroids
    const sums = Array.from({length:k}, ()=> Array(dim).fill(0));
    const counts = Array(k).fill(0);
    for(let i=0;i<n;i++){
      const c = labels[i]; counts[c]++;
      for(let j=0;j<dim;j++) sums[c][j] += data[i][j];
    }
    for(let c=0;c<k;c++){
      if(counts[c]===0){
        // reinit empty centroid
        centroids[c] = [...data[Math.floor(Math.random()*n)]];
      } else {
        centroids[c] = sums[c].map(v=> v/counts[c]);
      }
    }
    if(!moved) break;
  }
  return {labels, centroids};
}

// Standardize features (z-score)
function standardize(matrix){
  const cols = transpose(matrix);
  const means = cols.map(c=> mean(c));
  const sds = cols.map(c=> std(c) || 1);
  const out = matrix.map(row => row.map((v,j)=> (v - means[j]) / sds[j] ));
  return {data:out, means, sds};
}

// CSV utilities
function parseCSVFile(file, onComplete){
  Papa.parse(file, {
    header:true,
    dynamicTyping:true,
    skipEmptyLines:true,
    complete: function(results){
      onComplete(results.data);
    }
  });
}

function downloadCSV(rows, filename='labeled_data.csv'){
  const csv = Papa.unparse(rows);
  const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// Sample dataset generator (customer transactional features)
function genSample(n=300){
  const rows = [];
  for(let i=0;i<n;i++){
    // three archetypes: high-value, mid, low
    const r = Math.random();
    let recency, frequency, monetary, tenure;
    if(r<0.2){ // high value
      recency = Math.random()*30;
      frequency = 20 + Math.random()*60;
      monetary = 500 + Math.random()*2000;
      tenure = 24 + Math.random()*36;
    } else if(r<0.6){ // mid
      recency = 30 + Math.random()*120;
      frequency = 5 + Math.random()*30;
      monetary = 50 + Math.random()*500;
      tenure = 6 + Math.random()*48;
    } else { // low
      recency = 120 + Math.random()*365;
      frequency = Math.random()*5;
      monetary = 10 + Math.random()*200;
      tenure = Math.random()*12;
    }
    rows.push({CustomerID: `CUST_${i+1}`, Recency: Number(recency.toFixed(2)), Frequency: Number(frequency.toFixed(2)), Monetary: Number(monetary.toFixed(2)), Tenure: Number(tenure.toFixed(2))});
  }
  return rows;
}

// ---------- React component ----------
export default function SmartSeg(){
  const [table, setTable] = useState([]); // array of objects
  const [features, setFeatures] = useState(['Recency','Frequency','Monetary','Tenure']);
  const [selectedFeatures, setSelectedFeatures] = useState(['Recency','Frequency']);
  const [k, setK] = useState(3);
  const [labels, setLabels] = useState([]);
  const [centroids, setCentroids] = useState([]);
  const [pcaScores, setPcaScores] = useState([]);
  const [explained, setExplained] = useState([]);
  const [running, setRunning] = useState(false);
  const fileRef = useRef(null);

  const featureColumns = useMemo(()=>{
    if(table.length===0) return features;
    const keys = Object.keys(table[0]).filter(k=> typeof table[0][k] === 'number');
    return keys;
  }, [table]);

  function handleFile(e){
    const f = e.target.files[0];
    if(!f) return;
    parseCSVFile(f, (data)=>{
      setTable(data);
      // set default features
      const numeric = Object.keys(data[0]||{}).filter(k=> typeof data[0][k] === 'number');
      setSelectedFeatures(numeric.slice(0, Math.min(4,numeric.length)));
    });
  }

  function handleGenerateSample(){
    const rows = genSample(400);
    setTable(rows);
    setSelectedFeatures(['Recency','Frequency','Monetary']);
  }

  function runSegmentation(){
    if(table.length===0) return alert('No data: upload a CSV or generate sample.');
    if(selectedFeatures.length===0) return alert('Select at least one numeric feature.');
    setRunning(true);
    try{
      const matrix = table.map(r => selectedFeatures.map(f=> Number(r[f]||0)));
      const {data: zmatrix} = standardize(matrix);
      const km = kmeans(zmatrix, k, 200);
      setLabels(km.labels);
      setCentroids(km.centroids);
      // PCA for visualization
      const p = pca2(zmatrix);
      setPcaScores(p.scores);
      setExplained(p.explained);
    } catch(err){
      console.error(err); alert('Error during segmentation: '+err.message);
    } finally{ setRunning(false); }
  }

  const labeledRows = useMemo(()=>{
    if(table.length===0 || labels.length===0) return table;
    return table.map((r,i)=> ({...r, Segment: labels[i]}));
  }, [table, labels]);

  function downloadLabeled(){
    if(labeledRows.length===0) return alert('No labeled data to download.');
    downloadCSV(labeledRows,'smartseg_labeled.csv');
  }

  function segmentProfile(){
    if(labels.length===0) return [];
    const kcount = Math.max(...labels)+1;
    const profiles = [];
    for(let c=0;c<kcount;c++){
      const idx = labels.map((lab,i)=> lab===c ? i : -1).filter(i=> i>=0);
      const prof = {Segment:c, Count: idx.length};
      for(const f of selectedFeatures){
        const vals = idx.map(i=> Number(table[i][f] || 0));
        prof[`${f}_mean`] = vals.length? Number(mean(vals).toFixed(3)):0;
      }
      profiles.push(prof);
    }
    return profiles;
  }

  const profiles = segmentProfile();

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <motion.div initial={{opacity:0, y: -10}} animate={{opacity:1, y:0}} className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-semibold">SmartSeg</h1>
            <p className="text-sm text-gray-600">Interactive customer segmentation app — upload CSV or generate sample data.</p>
          </div>
          <div className="flex gap-3">
            <button className="px-4 py-2 bg-indigo-600 text-white rounded shadow" onClick={handleGenerateSample}>Generate Sample</button>
            <label className="px-4 py-2 bg-white border rounded shadow cursor-pointer">
              <input ref={fileRef} type="file" accept=".csv" onChange={handleFile} className="hidden" />
              Upload CSV
            </label>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="col-span-1 bg-white p-4 rounded shadow">
            <h2 className="font-semibold mb-2">Data & Features</h2>
            <div className="text-sm text-gray-600 mb-2">Rows: <strong>{table.length}</strong></div>
            <div className="mb-2">
              <div className="text-sm font-medium">Numeric features</div>
              <div className="flex flex-wrap gap-2 mt-2">
                {featureColumns.map(col=> (
                  <button key={col} onClick={()=>{
                    setSelectedFeatures(prev=> prev.includes(col) ? prev.filter(x=>x!==col) : [...prev, col]);
                  }} className={`px-3 py-1 rounded ${selectedFeatures.includes(col)? 'bg-indigo-600 text-white': 'bg-gray-100 text-gray-700'}`}>
                    {col}
                  </button>
                ))}
              </div>
            </div>
            <div className="mb-2">
              <label className="block text-sm">Number of clusters (k)</label>
              <input type="range" min="2" max="10" value={k} onChange={e=>setK(Number(e.target.value))} />
              <div className="text-sm">k = <strong>{k}</strong></div>
            </div>
            <div className="mt-4">
              <button disabled={running} onClick={runSegmentation} className="px-4 py-2 bg-green-600 text-white rounded shadow">{running? 'Running...':'Run Segmentation'}</button>
              <button onClick={downloadLabeled} className="ml-2 px-3 py-2 bg-gray-800 text-white rounded">Download CSV</button>
            </div>
          </div>

          <div className="col-span-2 bg-white p-4 rounded shadow">
            <h2 className="font-semibold mb-2">Visualization & Results</h2>
            <div style={{height:360}} className="mb-4">
              {pcaScores.length>0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid />
                    <XAxis type="number" dataKey="x" name="PC1" />
                    <YAxis type="number" dataKey="y" name="PC2" />
                    <Tooltip cursor={{strokeDasharray:'3 3'}} />
                    <Legend />
                    {Array.from(new Set(labels)).map(cluster => (
                      <Scatter key={cluster} name={`Segment ${cluster}`} data={pcaScores.map((s,i)=> ({x: s.x, y: s.y, segment: labels[i]})).filter(p=>p.segment===cluster)} fill={undefined} />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-400">Run segmentation to see PCA plot</div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-3 border rounded">
                <h3 className="font-medium mb-2">Cluster counts</h3>
                <ul className="text-sm space-y-1">
                  {labels.length>0 ? Array.from({length: Math.max(...labels)+1}, (_,c)=> (
                    <li key={c}>Segment {c}: <strong>{labels.filter(l=>l===c).length}</strong></li>
                  )) : <li>No results yet</li>}
                </ul>
              </div>

              <div className="p-3 border rounded overflow-auto">
                <h3 className="font-medium mb-2">Profiles (segment means)</h3>
                {profiles.length>0 ? (
                  <table className="w-full text-sm">
                    <thead>
                      <tr>
                        <th className="text-left">Segment</th>
                        <th className="text-left">Count</th>
                        {selectedFeatures.map(f=> <th key={f} className="text-left">{f} mean</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {profiles.map(p=> (
                        <tr key={p.Segment} className="border-t">
                          <td>{p.Segment}</td>
                          <td>{p.Count}</td>
                          {selectedFeatures.map(f=> <td key={f}>{p[`${f}_mean`]}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="text-sm text-gray-500">No segments computed yet.</div>
                )}
              </div>
            </div>

          </div>
        </div>

        <div className="mt-6 bg-white p-4 rounded shadow">
          <h2 className="font-semibold mb-2">Preview rows</h2>
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-100">
                <tr>
                  {Object.keys(table[0]||{}).map(col=> (<th key={col} className="p-2 text-left">{col}</th>))}
                  {labels.length>0 && <th className="p-2">Segment</th>}
                </tr>
              </thead>
              <tbody>
                {table.slice(0,10).map((r,i)=> (
                  <tr key={i} className="border-t">
                    {Object.keys(r).map(col=> (<td key={col} className="p-2">{String(r[col])}</td>))}
                    {labels.length>0 && <td className="p-2">{labels[i]}</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}
