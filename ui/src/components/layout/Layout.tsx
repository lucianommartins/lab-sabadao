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

import { useState } from 'react';
import { Box, Drawer, List, ListItem, ListItemIcon, ListItemText, AppBar, Toolbar, Typography, IconButton } from '@mui/material';
import { Menu, Activity, BarChart2, Settings } from 'lucide-react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';

const drawerWidth = 240;

/**
 * Main dashboard shell layout component.
 *
 * Renders top app bar, collapsible navigation drawer with view links,
 * and router Outlet container for active page content.
 */
export function Layout() {
  const [open, setOpen] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  const toggleDrawer = () => {
    setOpen(!open);
  };

  const menuItems = [
    { text: 'Analytics & Evals', icon: <Activity size={20} />, path: '/' },
    { text: 'Compare', icon: <BarChart2 size={20} />, path: '/compare' },
  ];

  return (
    <>
      <Box sx={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden' }}>
        <AppBar position="fixed" elevation={0} sx={{ zIndex: (theme) => theme.zIndex.drawer + 1, backgroundColor: '#ffffff', color: '#1a1a1a', borderBottom: '1px solid #e0e0e0' }}>
          <Toolbar variant="dense" sx={{ minHeight: 56 }}>
            <IconButton
              edge="start"
              color="inherit"
              aria-label="open drawer"
              onClick={toggleDrawer}
              sx={{ marginRight: 2 }}
            >
              <Menu size={20} />
            </IconButton>
            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1, fontWeight: 500, fontSize: '1.2rem', color: 'primary.main' }}>
              GBench
            </Typography>
            <IconButton color="inherit" size="small">
              <Settings size={20} />
            </IconButton>
          </Toolbar>
        </AppBar>
        
        <Drawer
          variant="persistent"
          anchor="left"
          open={open}
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: drawerWidth,
              boxSizing: 'border-box',
              borderRight: '1px solid #e0e0e0',
              backgroundColor: '#fafafa',
            },
          }}
        >
          <Toolbar variant="dense" sx={{ minHeight: 56 }} />
          <Box sx={{ overflow: 'auto', mt: 2 }}>
            <List>
              {menuItems.map((item) => (
                <ListItem 
                  key={item.text} 
                  disablePadding
                  sx={{
                    mb: 0.5,
                    px: 1,
                  }}
                >
                  <Box
                    onClick={() => navigate(item.path)}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      width: '100%',
                      p: 1.5,
                      borderRadius: 1,
                      cursor: 'pointer',
                      backgroundColor: location.pathname === item.path ? '#e8f0fe' : 'transparent',
                      color: location.pathname === item.path ? '#1a73e8' : '#5f6368',
                      '&:hover': {
                        backgroundColor: location.pathname === item.path ? '#e8f0fe' : '#f1f3f4',
                      }
                    }}
                  >
                    <ListItemIcon sx={{ color: 'inherit', minWidth: 40 }}>
                      {item.icon}
                    </ListItemIcon>
                    <ListItemText 
                      primary={
                        <Typography sx={{ 
                          fontSize: '0.95rem', 
                          fontWeight: location.pathname === item.path ? 600 : 500,
                        }}>
                          {item.text}
                        </Typography>
                      }
                    />
                  </Box>
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>
        
        <Box component="main" sx={{ 
          flexGrow: 1, 
          p: { xs: 2, md: 3 }, 
          backgroundColor: '#f8f9fa', 
          height: '100vh', 
          overflow: 'auto',
          marginLeft: open ? 0 : `-${drawerWidth}px`,
          transition: 'margin 0.3s'
        }}>
          <Toolbar variant="dense" sx={{ minHeight: 56 }} />
          <Box sx={{ width: '100%', px: 1 }}>
            <Outlet />
          </Box>
        </Box>
      </Box>
    </>
  );
}
