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

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  Typography, Box, Select, MenuItem,
  FormControl, InputLabel, Button, Paper,
  Checkbox, IconButton, CircularProgress, TextField, Dialog, DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

interface ScenarioNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: ScenarioNode[];
}

/**
 * New Evaluation job trigger modal and form component.
 *
 * Provides controls to configure models, format geometries, batch sizes,
 * and capability test scenarios, submitting jobs to `/api/runs`.
 */
export function NewEvaluation({ onClose, onRunTriggered }: { onClose?: () => void; onRunTriggered?: () => void } = {}) {
  const navigate = useNavigate();
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [scenarioTree, setScenarioTree] = useState<ScenarioNode[]>([]);
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [tags, setTags] = useState('');

  useEffect(() => {
    fetchModels();
    fetchScenarios();
  }, []);

  const fetchModels = async () => {
    try {
      const res = await axios.get('/api/models');
      setModels(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchScenarios = async () => {
    try {
      const res = await axios.get('/api/scenarios');
      setScenarioTree(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  const getAllFilePaths = (nodes: ScenarioNode[]): string[] => {
    const paths: string[] = [];
    const traverse = (node: ScenarioNode) => {
      if (node.type === 'file') paths.push(node.path);
      if (node.children) node.children.forEach(traverse);
    };
    nodes.forEach(traverse);
    return paths;
  };

  const handleRun = async () => {
    if (!selectedModel) return;
    setLoading(true);
    try {
      const parsedTags = tags.split(',').map(t => t.trim()).filter(t => t.length > 0);
      await axios.post('/api/runs', {
        model_id: selectedModel,
        format: 'hf',
        run_type: 'quality',
        preset: 'quick',
        selected_scenarios: selectedScenarios.length > 0 ? selectedScenarios : null,
        tags: parsedTags.length > 0 ? parsedTags : null
      });
      if (onRunTriggered) {
        onRunTriggered();
      } else {
        navigate('/');
      }
    } catch (e: any) {
      alert(e.response?.data?.detail || "Error starting run.");
    }
    setLoading(false);
  };

  const renderNode = (node: ScenarioNode) => {
    if (node.type === 'directory') {
      const allChildPaths = getAllFilePaths([node]);
      const selectedChildren = allChildPaths.filter(p => selectedScenarios.includes(p));
      const isAllChecked = selectedChildren.length === allChildPaths.length && allChildPaths.length > 0;
      const isSomeChecked = selectedChildren.length > 0 && !isAllChecked;

      return (
        <Box key={node.path} sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            <Checkbox
              size="small"
              checked={isAllChecked}
              indeterminate={isSomeChecked}
              onChange={() => {
                if (isAllChecked) {
                  setSelectedScenarios(prev => prev.filter(p => !allChildPaths.includes(p)));
                } else {
                  setSelectedScenarios(prev => {
                    const filtered = prev.filter(p => !allChildPaths.includes(p));
                    return [...filtered, ...allChildPaths];
                  });
                }
              }}
            />
            <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.primary', display: 'flex', alignItems: 'center', gap: 0.5 }}>
              📁 {node.name}
            </Typography>
          </Box>
          <Box sx={{ pl: 3, borderLeft: '1px dashed #e0e0e0', ml: 1.5 }}>
            {node.children && node.children.map(renderNode)}
          </Box>
        </Box>
      );
    } else {
      const isChecked = selectedScenarios.includes(node.path);
      return (
        <Box key={node.path} sx={{ display: 'flex', alignItems: 'center', py: 0.25 }}>
          <Checkbox
            size="small"
            checked={isChecked}
            onChange={() => {
              if (isChecked) {
                setSelectedScenarios(prev => prev.filter(p => p !== node.path));
              } else {
                setSelectedScenarios(prev => [...prev, node.path]);
              }
            }}
          />
          <Box sx={{ ml: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 400 }}>
              {node.name}
            </Typography>
            <Typography variant="caption" color="textSecondary" sx={{ fontSize: '0.7rem', display: 'block', mt: -0.25 }}>
              {node.path}
            </Typography>
          </Box>
        </Box>
      );
    }
  };

  const formContent = (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #e0e0e0', pb: 2 }}>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>New Evaluation</Typography>
        {onClose && (
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        )}
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {/* Target Model Selection */}
        <Box>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>Target Model</Typography>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
            <FormControl fullWidth size="medium">
              <InputLabel>Select Model</InputLabel>
              <Select
                value={selectedModel}
                label="Select Model"
                onChange={e => setSelectedModel(e.target.value)}
              >
                {models.map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
              </Select>
            </FormControl>
            <Button
              variant="outlined"
              size="large"
              sx={{ minWidth: 150, height: 56 }}
              onClick={() => setImportDialogOpen(true)}
            >
              Import model
            </Button>
          </Box>
        </Box>

        {/* Tags */}
        <Box>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>Tags (Optional)</Typography>
          <TextField
            fullWidth
            value={tags}
            onChange={e => setTags(e.target.value)}
            placeholder="e.g. baseline, test, regression"
            helperText="Comma separated tags to identify this run in the analytics chart."
          />
        </Box>

        {/* Scenarios Checklist */}
        <Box>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>Quality Scenarios</Typography>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2, flexWrap: 'wrap', gap: 1 }}>
            <Typography variant="body2" color="text.secondary" sx={{ maxWidth: '70%' }}>
              Select specific scenarios to run, or leave all unchecked to run the full comprehensive suite.
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button size="small" variant="outlined" onClick={() => setSelectedScenarios(getAllFilePaths(scenarioTree))}>
                Select All
              </Button>
              <Button size="small" variant="outlined" onClick={() => setSelectedScenarios([])}>
                Clear
              </Button>
            </Box>
          </Box>

          {/* Scrollable Tree Container */}
          <Box sx={{ maxHeight: onClose ? 450 : 350, overflow: 'auto', border: '1px solid #e0e0e0', borderRadius: 2, p: 2, bgcolor: '#f8f9fa' }}>
            {scenarioTree.map(renderNode)}
          </Box>
        </Box>
      </Box>

      {/* Footer Actions */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, pt: 3, borderTop: '1px solid #e0e0e0', mt: 2 }}>
        <Button onClick={onClose || (() => navigate('/'))} disabled={loading}>Cancel</Button>
        <Button
          variant="contained"
          size="large"
          disabled={!selectedModel || loading}
          onClick={handleRun}
          sx={{ minWidth: 150 }}
        >
          {loading ? <CircularProgress size={24} color="inherit" /> : "Start Benchmark"}
        </Button>
      </Box>
    </Box>
  );

  const importDialog = (
    <Dialog
      open={importDialogOpen}
      onClose={() => setImportDialogOpen(false)}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle sx={{ m: 0, p: 2, fontWeight: 600 }}>
        Import a model
        <IconButton
          onClick={() => setImportDialogOpen(false)}
          sx={{
            position: 'absolute',
            right: 8,
            top: 8,
            color: (theme) => theme.palette.grey[500],
          }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <Typography sx={{ mb: 2 }} color="text.secondary">
          To benchmark a model the service cannot download directly, stage its weights to your GCS bucket
          from a machine that can reach the source. It will automatically appear in the model list once complete.
        </Typography>
        <Box sx={{ bgcolor: 'grey.100', p: 2, borderRadius: 1, fontFamily: 'monospace', wordBreak: 'break-all', fontSize: '0.85rem', color: 'primary.main' }}>
          gbench --models &lt;hf-model-id&gt; --stage-to-gcs gs://YOUR_MODELS_BUCKET/staging/
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setImportDialogOpen(false)} variant="contained">Close</Button>
      </DialogActions>
    </Dialog>
  );

  if (onClose) {
    return (
      <Box>
        {formContent}
        {importDialog}
      </Box>
    );
  }

  return (
    <Box sx={{ p: 4 }}>
      <Paper sx={{ p: 4, maxWidth: 800, mx: 'auto', border: '1px solid #e0e0e0', boxShadow: 'none', borderRadius: 2 }}>
        {formContent}
      </Paper>
      {importDialog}
    </Box>
  );
}
