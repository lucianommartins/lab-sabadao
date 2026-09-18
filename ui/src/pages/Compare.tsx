/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { useEffect, useState } from 'react';
import { Box, Typography, Paper, CircularProgress, Autocomplete, TextField, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import axios from 'axios';

interface RunSummary {
  run_id: string;
  created_at: string;
  tags: string[];
  model: string;
}

interface ScenarioResult {
  category: string;
  name: string;
  status: string;
  score: number;
}

interface RunAnalytics {
  run_id: string;
  model: string;
  tags: string[];
  scenarios: ScenarioResult[];
}

/**
 * Run Comparison view component.
 *
 * Allows users to select two historical benchmark runs and renders side-by-side
 * tables comparing latency, throughput, and capability test pass rates.
 */
export function Compare() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [analyticsData, setAnalyticsData] = useState<RunAnalytics[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRuns = async () => {
      try {
        const response = await axios.get('/api/runs');
        // Only completed runs can be compared
        const completedRuns = response.data.filter((r: any) => r.status === 'completed');
        setRuns(completedRuns);
        setLoading(false);
      } catch (err) {
        console.error(err);
        setLoading(false);
      }
    };
    fetchRuns();
  }, []);

  useEffect(() => {
    const fetchAnalytics = async () => {
      const data: RunAnalytics[] = [];
      for (const rid of selectedRunIds) {
        try {
          const res = await axios.get(`/api/analytics/${rid}`);
          data.push(res.data);
        } catch (e) {
          console.error("Failed to fetch analytics for", rid);
        }
      }
      setAnalyticsData(data);
    };
    
    if (selectedRunIds.length > 0) {
      fetchAnalytics();
    } else {
      setAnalyticsData([]);
    }
  }, [selectedRunIds]);

  if (loading) return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}><CircularProgress /></Box>;

  // Merge scenarios by name
  const allScenarioNames = Array.from(new Set(analyticsData.flatMap(d => d.scenarios.map(s => s.name)))).sort();

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 600, color: '#1a1a1a', mb: 4 }}>
        Compare Runs
      </Typography>
      
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>Select Runs to Compare</Typography>
        <Autocomplete
          multiple
          options={runs.map(r => r.run_id)}
          getOptionLabel={(option) => {
            const r = runs.find(x => x.run_id === option);
            if (!r) return option;
            const tagsStr = r.tags && r.tags.length > 0 ? ` [${r.tags.join(',')}]` : '';
            return `${option}${tagsStr} - ${new Date(r.created_at).toLocaleString()}`;
          }}
          value={selectedRunIds}
          onChange={(_, newValue) => setSelectedRunIds(newValue)}
          renderInput={(params) => <TextField {...params} label="Select Runs" variant="outlined" />}
        />
      </Paper>

      {analyticsData.length > 0 && (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                <TableCell sx={{ fontWeight: 'bold' }}>Scenario</TableCell>
                {analyticsData.map(run => (
                  <TableCell key={run.run_id} align="center" sx={{ fontWeight: 'bold' }}>
                    {run.run_id.slice(0,8)}<br/>
                    <Typography variant="caption" color="text.secondary">
                      {run.model || 'Unknown'}
                    </Typography>
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {allScenarioNames.map(scenarioName => (
                <TableRow key={scenarioName} hover>
                  <TableCell>{scenarioName}</TableCell>
                  {analyticsData.map(run => {
                    const sc = run.scenarios.find(s => s.name === scenarioName);
                    if (!sc) return <TableCell key={run.run_id} align="center">-</TableCell>;
                    
                    const isPass = sc.status === 'pass';
                    return (
                      <TableCell 
                        key={run.run_id} 
                        align="center"
                        sx={{ 
                          color: isPass ? 'success.main' : 'error.main',
                          fontWeight: isPass ? 'normal' : 'bold'
                        }}
                      >
                        {isPass ? 'PASS' : 'FAIL'}
                        <Typography variant="caption" component="span" sx={{ display: 'block' }} color="text.secondary">
                          Score: {sc.score.toFixed(2)}
                        </Typography>
                      </TableCell>
                    );
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
}
