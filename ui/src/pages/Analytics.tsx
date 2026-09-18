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
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  TextField,
  Autocomplete,
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import axios from 'axios';

interface ScenarioSummary {
  name: string;
  path: string;
  status: string;
}

interface PerformanceMetrics {
  ttft_p50_ms?: number;
  ttft_mean_ms?: number;
  tpot_p50_ms?: number;
  tpot_mean_ms?: number;
  output_throughput?: number;
  request_throughput?: number;
  throughput_output_tok_s?: number;
  throughput_total_tok_s?: number;
}

interface RunSummary {
  run_id: string;
  created_at: string;
  tags: string[];
  model: string;
  pass_rate: number;
  total_scenarios: number;
  scenarios?: ScenarioSummary[];
  // Serving/throughput pillar metrics from /api/analytics (latency in ms, throughput in tokens/s),
  // so a run surfaces performance alongside its quality pass rate.
  performance?: PerformanceMetrics;
}

interface ScenarioStability {
  name: string;
  path: string;
  history: ('pass' | 'fail')[];
  failCount: number;
  totalCount: number;
}

/**
 * Historical and live analytics charts component.
 *
 * Renders a Recharts pass-rate trend over time across evaluated runs. Each run also carries
 * serving/throughput performance metrics (latency and throughput) in `performance`, provided by the
 * /api/analytics endpoints for surfacing alongside the quality pass rate.
 */
export function Analytics({ liveProgress }: { liveProgress?: any }) {
  const [data, setData] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('/api/analytics/summary');
        // Sort chronologically
        const sorted = response.data.sort((a: RunSummary, b: RunSummary) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        );
        setData(sorted);
        setLoading(false);
      } catch (err: any) {
        setError(err.message);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}><CircularProgress /></Box>;
  if (error) return <Typography color="error">Error: {error}</Typography>;

  // Extract unique models and tags for the filters
  const uniqueModels = Array.from(new Set(data.map(d => d.model)));
  const uniqueTags = Array.from(new Set(data.flatMap(d => d.tags)));

  // Filter data based on selections
  const filteredData = data.filter(d => {
    if (selectedModels.length > 0 && !selectedModels.includes(d.model)) return false;
    if (selectedTags.length > 0 && !selectedTags.some(t => d.tags.includes(t))) return false;
    return true;
  });

  // 1. Compute Metrics
  const totalRuns = filteredData.length;
  const peakPassRate = filteredData.length > 0
    ? Math.max(...filteredData.map(d => d.pass_rate))
    : 0;

  const recentRuns = filteredData.slice(-5);
  const recentAvgPassRate = recentRuns.length > 0
    ? recentRuns.reduce((acc, curr) => acc + curr.pass_rate, 0) / recentRuns.length
    : 0;


  const lowestPassRate = filteredData.length > 0
    ? Math.min(...filteredData.map(d => d.pass_rate))
    : 0;


  // 2. Format data for chart
  const chartData = filteredData.map(d => ({
    name: new Date(d.created_at).toLocaleDateString() + ' ' + new Date(d.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    pass_rate: d.pass_rate,
    run_id: d.run_id,
    model: d.model,
    tags: d.tags.join(', ')
  }));

  // 3. Compute Scenario Stability Diagnostics (based on chronologically ordered filtered data)
  const scenarioMap: Record<string, { name: string; history: { runId: string; status: 'pass' | 'fail' }[] }> = {};
  filteredData.forEach(run => {
    (run.scenarios || []).forEach((s) => {
      const key = s.name;
      if (!scenarioMap[key]) {
        scenarioMap[key] = { name: s.name, history: [] };
      }
      scenarioMap[key].history.push({
        runId: run.run_id,
        status: s.status === 'pass' ? 'pass' : 'fail'
      });
    });
  });

  const stabilityList: ScenarioStability[] = Object.entries(scenarioMap).map(([, info]) => {
    // Take the last 10 runs of history
    const recentHistory = info.history.slice(-10);
    const failCount = recentHistory.filter(h => h.status === 'fail').length;
    return {
      name: info.name,
      path: '',
      history: recentHistory.map(h => h.status),
      failCount,
      totalCount: recentHistory.length
    };
  });

  // Sort by failure rate descending (worst/most unstable first)
  stabilityList.sort((a, b) => {
    const aRate = a.totalCount ? a.failCount / a.totalCount : 0;
    const bRate = b.totalCount ? b.failCount / b.totalCount : 0;
    if (bRate !== aRate) return bRate - aRate;
    return b.totalCount - a.totalCount; // tiebreaker: more runs first
  });

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {/* Header Filters Row */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
        <Typography variant="h5" sx={{ fontWeight: 600, color: 'text.primary' }}>Analytics Dashboard</Typography>
        <Box sx={{ display: 'flex', gap: 2, minWidth: 400 }}>
          <Autocomplete
            multiple
            size="small"
            options={uniqueModels}
            value={selectedModels}
            onChange={(_, newValue) => setSelectedModels(newValue)}
            renderInput={(params) => <TextField {...params} label="Model Filter" placeholder="All" />}
            sx={{ flex: 1 }}
          />
          <Autocomplete
            multiple
            size="small"
            options={uniqueTags}
            value={selectedTags}
            onChange={(_, newValue) => setSelectedTags(newValue)}
            renderInput={(params) => <TextField {...params} label="Tag Filter" placeholder="All" />}
            sx={{ flex: 1 }}
          />
        </Box>
      </Box>

      {/* Metric Cards Grid */}
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card sx={{ border: '1px solid #e0e0e0', boxShadow: 'none', borderRadius: 2 }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Typography color="text.secondary" variant="overline" sx={{ fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.04em', lineHeight: 1.2, display: 'block' }}>Total Runs</Typography>
              <Typography sx={{ fontWeight: 700, mt: 0.25, fontSize: '1.3rem', lineHeight: 1.2 }}>{totalRuns}</Typography>
              <Typography color="text.secondary" sx={{ fontSize: '0.7rem', mt: 0.25, display: 'block', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>Evaluations matched</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card sx={{ border: '1px solid #e0e0e0', boxShadow: 'none', borderRadius: 2 }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Typography color="text.secondary" variant="overline" sx={{ fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.04em', lineHeight: 1.2, display: 'block' }}>Recent Avg (Last 5)</Typography>
              <Typography sx={{ fontWeight: 700, mt: 0.25, fontSize: '1.3rem', lineHeight: 1.2, color: 'info.main' }}>
                {Math.round(recentAvgPassRate * 100)}%
              </Typography>
              <Typography color="text.secondary" sx={{ fontSize: '0.7rem', mt: 0.25, display: 'block', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>Last 5 runs average score</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card sx={{ border: '1px solid #e0e0e0', boxShadow: 'none', borderRadius: 2 }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Typography color="text.secondary" variant="overline" sx={{ fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.04em', lineHeight: 1.2, display: 'block' }}>Peak Pass Rate</Typography>
              <Typography sx={{ fontWeight: 700, mt: 0.25, fontSize: '1.3rem', lineHeight: 1.2, color: 'success.main' }}>
                {Math.round(peakPassRate * 100)}%
              </Typography>
              <Typography color="text.secondary" sx={{ fontSize: '0.7rem', mt: 0.25, display: 'block', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>Best score globally</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card sx={{ border: '1px solid #e0e0e0', boxShadow: 'none', borderRadius: 2 }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Typography color="text.secondary" variant="overline" sx={{ fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.04em', lineHeight: 1.2, display: 'block' }}>Worst Pass Rate</Typography>
              <Typography sx={{ fontWeight: 700, mt: 0.25, fontSize: '1.3rem', lineHeight: 1.2, color: lowestPassRate < 0.5 ? 'error.main' : 'warning.main' }}>
                {Math.round(lowestPassRate * 100)}%
              </Typography>
              <Typography color="text.secondary" sx={{ fontSize: '0.7rem', mt: 0.25, display: 'block', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>Lowest score globally</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Live Run Progress */}
      {liveProgress && (
        <Paper 
          sx={{ 
            p: 3, 
            mb: 3, 
            border: '1px solid #1976d2', 
            bgcolor: '#f4f9ff', 
            borderRadius: 2,
            boxShadow: 'none',
            display: 'flex',
            flexDirection: 'column',
            gap: 2
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box>
              <Typography variant="overline" color="primary.main" sx={{ fontWeight: 700, fontSize: '0.65rem', letterSpacing: '0.08em' }}>
                Active Evaluation Run Progress
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 700, mt: 0.25 }}>
                {liveProgress.modelName}
              </Typography>
              <Typography variant="caption" color="textSecondary" sx={{ fontFamily: 'monospace', display: 'block', mt: 0.25 }}>
                {liveProgress.runId}
              </Typography>
            </Box>
            <CircularProgress size={24} sx={{ color: 'primary.main' }} />
          </Box>
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2, mt: 1 }}>
            <Box sx={{ display: 'flex', gap: 4 }}>
              <Box>
                <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, display: 'block' }}>SCENARIOS RUN</Typography>
                <Typography variant="h5" sx={{ fontWeight: 700 }}>
                  {liveProgress.passed + liveProgress.failed} / {liveProgress.total || '?'}
                </Typography>
              </Box>
              <Box>
                <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, display: 'block' }}>PASSED</Typography>
                <Typography variant="h5" sx={{ fontWeight: 700, color: 'success.main' }}>
                  {liveProgress.passed}
                </Typography>
              </Box>
              <Box>
                <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, display: 'block' }}>FAILED</Typography>
                <Typography variant="h5" sx={{ fontWeight: 700, color: 'error.main' }}>
                  {liveProgress.failed}
                </Typography>
              </Box>
              <Box>
                <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, display: 'block' }}>PASS RATE</Typography>
                <Typography variant="h5" sx={{ fontWeight: 700, color: 'primary.main' }}>
                  {liveProgress.passed + liveProgress.failed > 0 
                    ? Math.round(liveProgress.passed / (liveProgress.passed + liveProgress.failed) * 100) 
                    : 0}%
                </Typography>
              </Box>
            </Box>
          </Box>

          <Box sx={{ width: '100%', mt: 0.5 }}>
            <LinearProgress 
              variant="determinate" 
              value={liveProgress.total > 0 ? ((liveProgress.passed + liveProgress.failed) / liveProgress.total) * 100 : 0} 
              sx={{ height: 6, borderRadius: 3, bgcolor: 'rgba(25, 118, 210, 0.1)' }}
            />
          </Box>

          {liveProgress.completedScenarios.length > 0 && (
            <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic', display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
              <span>Latest:</span>
              <span>{liveProgress.completedScenarios[liveProgress.completedScenarios.length - 1].passed ? '✅' : '❌'}</span>
              <span style={{ fontWeight: 500 }}>{liveProgress.completedScenarios[liveProgress.completedScenarios.length - 1].scenario}</span>
            </Typography>
          )}

          {liveProgress.logs && liveProgress.logs.length > 0 && (
            <Box 
              sx={{ 
                mt: 2, 
                p: 1.5, 
                bgcolor: '#1e1e1e', 
                color: '#39ff14', 
                borderRadius: 1, 
                fontFamily: 'monospace', 
                fontSize: '0.7rem', 
                maxHeight: 120, 
                overflowY: 'auto',
                border: '1px solid #333',
                lineHeight: 1.3,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all'
              }}
            >
              {liveProgress.logs.map((log: string, idx: number) => (
                <div key={idx} style={{ paddingBottom: 2 }}>
                  {log}
                </div>
              ))}
            </Box>
          )}
        </Paper>
      )}

      {/* Line Chart Panel */}
      <Paper sx={{ p: 4, mb: 3, border: '1px solid #e0e0e0', boxShadow: 'none', borderRadius: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>Pass Rate Over Time</Typography>
        {chartData.length === 0 ? (
          <Box sx={{ p: 4, backgroundColor: '#f8f9fa', borderRadius: 2, textAlign: 'center', border: '1px dashed #ccc' }}>
            <Typography variant="body1" color="text.secondary">
              No benchmarks match your filters. Run an evaluation to see metrics here.
            </Typography>
          </Box>
        ) : (
          <Box sx={{ height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#666' }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 1]} tickFormatter={(val) => `${Math.round(val * 100)}%`} axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#666' }} />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}
                  formatter={(value: any, _name: any, props: any) => [`${Math.round((value as number) * 100)}%`, `Pass Rate (${props.payload.model})`]}
                  labelFormatter={(label, payload) => payload && payload.length > 0 ? `${label} (Run: ${payload[0].payload.run_id.slice(0, 8)})` : label}
                />
                <Line type="monotone" dataKey="pass_rate" stroke="#1a73e8" activeDot={{ r: 6 }} strokeWidth={3} dot={{ r: 4, strokeWidth: 0, fill: '#1a73e8' }} name="Pass Rate" />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        )}
      </Paper>

      {/* Stability Diagnostics Table */}
      <Paper sx={{ p: 4, border: '1px solid #e0e0e0', boxShadow: 'none', borderRadius: 2, display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ mb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>🔥 Instability Diagnostics</Typography>
          <Typography variant="body2" color="text.secondary">
            Scenarios sorted by failure rate (last 10 runs)
          </Typography>
        </Box>

        {stabilityList.length === 0 ? (
          <Box sx={{ p: 4, backgroundColor: '#f8f9fa', borderRadius: 2, textAlign: 'center', border: '1px dashed #ccc' }}>
            <Typography variant="body1" color="text.secondary">No scenario run data found.</Typography>
          </Box>
        ) : (
          <TableContainer sx={{ overflowY: 'auto', maxHeight: 350 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600, bgcolor: '#fff' }}>Scenario Name</TableCell>
                  <TableCell sx={{ fontWeight: 600, bgcolor: '#fff' }} align="center">History (Last 10)</TableCell>
                  <TableCell sx={{ fontWeight: 600, bgcolor: '#fff' }} align="right">Fail Rate</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {stabilityList.map((s, idx) => {
                  const failPercentage = s.totalCount ? Math.round((s.failCount / s.totalCount) * 100) : 0;
                  return (
                    <TableRow key={idx} hover sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                      <TableCell sx={{ fontWeight: 500, fontSize: '0.85rem' }}>
                        {s.name}
                      </TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: 'inline-flex', gap: 0.5, justifyContent: 'center', alignItems: 'center' }}>
                          {s.history.map((status, hIdx) => (
                            <Box
                              key={hIdx}
                              sx={{
                                width: 16,
                                height: 16,
                                borderRadius: '3px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                bgcolor: status === 'pass' ? 'success.main' : 'error.main',
                                color: '#fff',
                                fontSize: '0.55rem',
                                fontWeight: 'bold',
                                userSelect: 'none'
                              }}
                              title={status === 'pass' ? 'Passed' : 'Failed'}
                            >
                              {status === 'pass' ? 'P' : 'F'}
                            </Box>
                          ))}
                        </Box>
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 600, fontSize: '0.85rem', color: s.failCount > 0 ? 'error.main' : 'success.main', whiteSpace: 'nowrap' }}>
                        {failPercentage}% ({s.failCount}/{s.totalCount})
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Paper>
    </Box>
  );
}
