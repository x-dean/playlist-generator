import React from 'react';
import { AppShell, Navbar, Header, Text, NavLink, Group, Title } from '@mantine/core';
import { IconMusic, IconPlaylist, IconAnalyze, IconDashboard } from '@tabler/icons-react';
import { useLocation, useNavigate } from 'react-router-dom';

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const navigate = useNavigate();

  const navItems = [
    { icon: IconDashboard, label: 'Dashboard', path: '/' },
    { icon: IconMusic, label: 'Library', path: '/library' },
    { icon: IconAnalyze, label: 'Analysis', path: '/analysis' },
    { icon: IconPlaylist, label: 'Playlists', path: '/playlists' },
  ];

  return (
    <AppShell
      navbar={{
        width: 250,
        breakpoint: 'sm',
      }}
      header={{ height: 60 }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md">
          <Title order={3}>Playlista v2</Title>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            active={location.pathname === item.path}
            label={item.label}
            leftSection={<item.icon size="1rem" />}
            onClick={() => navigate(item.path)}
          />
        ))}
      </AppShell.Navbar>

      <AppShell.Main>{children}</AppShell.Main>
    </AppShell>
  );
}
