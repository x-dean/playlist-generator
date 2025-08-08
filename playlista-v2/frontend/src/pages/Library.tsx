import React from 'react';
import { Container, Title, Button, Table, Badge, Text, Group } from '@mantine/core';
import { IconScan, IconAnalyze } from '@tabler/icons-react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';

export function Library() {
  const { data: tracks, refetch } = useQuery({
    queryKey: ['tracks'],
    queryFn: () => api.get('/library/tracks?limit=50').then(res => res.data)
  });

  const handleScanLibrary = async () => {
    try {
      await api.post('/library/tracks/scan?path=C:\\Users\\Dean\\Documents\\test_music');
      refetch();
    } catch (error) {
      console.error('Scan failed:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'analyzed': return 'green';
      case 'analyzing': return 'blue';
      case 'failed': return 'red';
      default: return 'gray';
    }
  };

  return (
    <Container size="xl">
      <Group justify="space-between" mb="xl">
        <Title order={1}>Music Library</Title>
        <Group>
          <Button leftSection={<IconScan size="1rem" />} onClick={handleScanLibrary}>
            Scan Library
          </Button>
          <Button leftSection={<IconAnalyze size="1rem" />} variant="outline">
            Analyze All
          </Button>
        </Group>
      </Group>

      <Table striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Title</Table.Th>
            <Table.Th>Artist</Table.Th>
            <Table.Th>Album</Table.Th>
            <Table.Th>Duration</Table.Th>
            <Table.Th>BPM</Table.Th>
            <Table.Th>Key</Table.Th>
            <Table.Th>Status</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {tracks?.map((track: any) => (
            <Table.Tr key={track.id}>
              <Table.Td>{track.title}</Table.Td>
              <Table.Td>{track.artist}</Table.Td>
              <Table.Td>{track.album || 'N/A'}</Table.Td>
              <Table.Td>
                {track.duration ? `${Math.floor(track.duration / 60)}:${Math.floor(track.duration % 60).toString().padStart(2, '0')}` : 'N/A'}
              </Table.Td>
              <Table.Td>{track.bpm?.toFixed(0) || 'N/A'}</Table.Td>
              <Table.Td>{track.key || 'N/A'}</Table.Td>
              <Table.Td>
                <Badge color={getStatusColor(track.status)} size="sm">
                  {track.status}
                </Badge>
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>

      {(!tracks || tracks.length === 0) && (
        <Text ta="center" c="dimmed" mt="xl">
          No tracks found. Click "Scan Library" to discover music files.
        </Text>
      )}
    </Container>
  );
}
