import React from 'react';
import { Container, Title, Button, Table, Badge, Text, Group, Modal } from '@mantine/core';
import { IconPlus } from '@tabler/icons-react';
import { useQuery } from '@tanstack/react-query';
import { useDisclosure } from '@mantine/hooks';
import { api } from '../api/client';

export function Playlists() {
  const [opened, { open, close }] = useDisclosure(false);
  
  const { data: playlists, refetch } = useQuery({
    queryKey: ['playlists'],
    queryFn: () => api.get('/playlists/').then(res => res.data)
  });

  const handleGeneratePlaylist = async (method: string) => {
    try {
      await api.post(`/playlists/generate?method=${method}&size=25`);
      refetch();
      close();
    } catch (error) {
      console.error('Failed to generate playlist:', error);
    }
  };

  return (
    <Container size="xl">
      <Group justify="space-between" mb="xl">
        <Title order={1}>Playlists</Title>
        <Button leftSection={<IconPlus size="1rem" />} onClick={open}>
          Generate Playlist
        </Button>
      </Group>

      <Table striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Name</Table.Th>
            <Table.Th>Method</Table.Th>
            <Table.Th>Tracks</Table.Th>
            <Table.Th>Duration</Table.Th>
            <Table.Th>Avg BPM</Table.Th>
            <Table.Th>Created</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {playlists?.map((playlist: any) => (
            <Table.Tr key={playlist.id}>
              <Table.Td>{playlist.name}</Table.Td>
              <Table.Td>
                <Badge variant="light">{playlist.generation_method}</Badge>
              </Table.Td>
              <Table.Td>{playlist.track_count}</Table.Td>
              <Table.Td>
                {playlist.total_duration ? 
                  `${Math.floor(playlist.total_duration / 60)}m` : 'N/A'}
              </Table.Td>
              <Table.Td>
                {playlist.avg_bpm?.toFixed(0) || 'N/A'}
              </Table.Td>
              <Table.Td>
                {new Date(playlist.created_at).toLocaleDateString()}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>

      {(!playlists || playlists.length === 0) && (
        <Text ta="center" c="dimmed" mt="xl">
          No playlists yet. Generate your first playlist!
        </Text>
      )}

      <Modal opened={opened} onClose={close} title="Generate Playlist">
        <Group>
          <Button onClick={() => handleGeneratePlaylist('kmeans')}>
            K-Means Clustering
          </Button>
          <Button onClick={() => handleGeneratePlaylist('energy_flow')}>
            Energy Flow
          </Button>
          <Button onClick={() => handleGeneratePlaylist('harmonic_mixing')}>
            Harmonic Mixing
          </Button>
        </Group>
      </Modal>
    </Container>
  );
}
